import torch

def ctc_loss(output, target):
    ctc = torch.nn.CTCLoss()
    output_lenght = 

