import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from models.m02_ISIBrnoAIMT_BagOfWords.model import NN
from models.m02_ISIBrnoAIMT_BagOfWords.train import *
from models.m02_ISIBrnoAIMT_BagOfWords.dataset import PtbXlDataset
import logging

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def setup_logging():
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')


def main():
    n_BoW = 20
    dataset = PtbXlDataset('data_ptb-xl/', 'train', n_BoW)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)

    val_dataset = PtbXlDataset('data_ptb-xl/', 'val', n_BoW)
    val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=True, num_workers=4)

    model = NN(n_BoW).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        logger.info("DataParallel is used")
        model = nn.DataParallel(model)
        logger.info("DataParallel is used correctly")

    train(model, dataloader, val_dataloader, criterion, optimizer, n_epochs=500)


if __name__ == '__main__':
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting the program")

    main()
