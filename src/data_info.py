import os
from datapr import PreprocessedLibriSpeech
from torch.utils.data import DataLoader
import torch


def collate_fn(batch):
    specs, labels = zip(*batch)

    specs = torch.stack(specs)  # [batch, 80, 157]

    label_lengths = torch.tensor([len(label) for label in labels])
    labels = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=0  # Use 0 as padding index
    )

    return specs, labels, label_lengths


try:
    dataset = PreprocessedLibriSpeech(
        root="../input", url="test-clean", augment=False, download=True
    )

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    for mel_specs, labels, label_lengths in dataloader:
        print(f"Spectrograms shape: {mel_specs.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Label lengths: {label_lengths}")
        break

except Exception as e:
    print(f"Error loading dataset: {e}")
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
    print("Current directory:", current_dir)
    print(
        "Contents of input directory:",
        os.listdir(os.path.join(current_dir, "../input")),
    )
