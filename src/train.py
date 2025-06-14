from math import log
import torch
from torch.utils.data import DataLoader
from model import ASRModel
from datapr import PreprocessedLibriSpeech
import torch.nn as nn
from tqdm import tqdm
import argparse
import wandb


def collate_fn(batch):
    specs, labels = zip(*batch)

    specs = torch.stack(specs)  # [batch, 80, 157]

    label_lengths = torch.tensor([len(label) for label in labels])
    labels = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=0  # Use 0 as padding index
    )

    return specs, labels, label_lengths


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for spectrograms, labels, label_lengths in tqdm(dataloader, desc="Training"):
        spectrograms = spectrograms.to(device)
        log_probs = model(spectrograms)

        input_legths = torch.full(
            size=(log_probs.size(0),), fill_value=log_probs.size(1), dtype=torch.long
        ).to(device)

        loss = criterion(
            log_probs.permute(1, 0, 2),
            labels.to(device),
            input_legths,
            label_lengths.to(device),
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)  # Gradient clipping
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def greedy_decode(log_probs, char2idx):
    """Convert network outputs to text using greedy decoding"""
    idx2char = {v: k for k, v in char2idx.items()}

    # Get most probable characters at each time step
    _, max_indices = torch.max(log_probs, dim=2)  # [batch, time]
    max_indices = max_indices.cpu().numpy()

    decoded_texts = []
    for sample in max_indices:
        # Merge repeated characters
        merged = []
        prev_char = None
        for char_idx in sample:
            if char_idx != prev_char:
                merged.append(char_idx)
                prev_char = char_idx

        # Remove blank tokens
        merged = [idx2char[c] for c in merged if c != char2idx["-"]]
        decoded_texts.append("".join(merged))

    return decoded_texts


def calculate_wer(reference, hypothesis):
    """Compute Word Error Rate between reference and hypothesis texts"""
    ref_words = reference.split()
    hyp_words = hypothesis.split()

    # Build distance matrix
    d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]

    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j

    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitution = d[i - 1][j - 1] + 1
                insertion = d[i][j - 1] + 1
                deletion = d[i - 1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    errors = d[-1][-1]
    total_words = len(ref_words)

    return errors / max(total_words, 1), errors, total_words


def evaluate(model, dataloader, char2idx, device):
    model.eval()
    total_wer = 0
    total_samples = 0

    with torch.no_grad():
        for spectrograms, labels, label_lengths in tqdm(dataloader, desc="Evaluating"):
            spectrograms = spectrograms.to(device)

            # Get predictions
            log_probs = model(spectrograms)
            predictions = greedy_decode(log_probs, char2idx)
            idx2char = {v: k for k, v in char2idx.items()}
            references = [
                "".join([idx2char[c.item()] for c in labels[start:end]])
                for start, end in zip(
                    [0] + label_lengths.cumsum(dim=0).tolist()[:-1],
                    label_lengths.cumsum(dim=0).tolist(),
                )
            ]
            for ref, pred in zip(references, predictions):
                wer, _, _ = calculate_wer(ref, pred)
                total_wer += wer
                total_samples += 1

    return total_wer / total_samples


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = PreprocessedLibriSpeech(
        root="../input", url="test-clean", augment=False, download=True
    )
    valid_dataset = PreprocessedLibriSpeech(
        root="../input", url="dev-clean", augment=False, download=True
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size, collate_fn=collate_fn
    )

    model = ASRModel(num_classes=len(train_dataset.char2idx)).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ctc_loss = nn.CTCLoss(blank=train_dataset.char2idx["-"], zero_infinity=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=3)

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, ctc_loss, device)
        valid_wer = evaluate(model, valid_loader, train_dataset.char2idx, device)

        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"Train Loss: {train_loss:.4f} | Valid WER: {valid_wer:.2%}")

        scheduler.step(valid_wer)
        best_wer = float("inf")
        # Save checkpoint
        if valid_wer < best_wer:
            torch.save(model.state_dict(), f"best_model_{valid_wer:.2f}.pt")
            best_wer = valid_wer


if __name__ == "__main__":
    main()
