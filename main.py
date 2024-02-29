import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from models.m02_ISIBrnoAIMT_BagOfWords.model import NN
from models.m02_ISIBrnoAIMT_BagOfWords.train import *
from models.m02_ISIBrnoAIMT_BagOfWords.dataset import PtbXlDataset


DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def main():
    n_BoW = 20
    dataset = PtbXlDataset('data_ptb-xl/', 'train', n_BoW)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    val_dataset = PtbXlDataset('data_ptb-xl/', 'val', n_BoW)
    val_dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = NN(n_BoW).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    train(model, dataloader, val_dataloader, criterion, optimizer, n_epochs=500)


if __name__ == '__main__':
    main()
