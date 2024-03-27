import os
import time
import logging
import numpy as np
import torch
import tqdm
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, jaccard_score


DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
logger = logging.getLogger(__name__)

def train(model, dataloader, val_dataloader, criterion, optimizer, n_epochs=10, patience=5):
    logger.info("Start training")
    best_score = 0.0
    early_stopping_counter = 0

    for epoch in range(n_epochs):
        start_time = time.time()
        model.train(True)
        losses = []
        for signals, reports in dataloader:
            signals = signals.unsqueeze(2).to(DEVICE).float()
            reports = reports.to(DEVICE).float()
            leads = torch.ones(signals.shape[0], signals.shape[1]).to(DEVICE).float()

            optimizer.zero_grad()

            y, p = model(signals, leads)

            loss = criterion(y, reports)

            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        mean_losses = np.mean(losses)
        f1, iou = get_metrics(model, val_dataloader)

        if iou > best_score:
            best_score = iou
            early_stopping_counter = 0
            n_bow = reports.shape[1]
            torch.save(model.state_dict(), f'./models/m02_ISIBrnoAIMT_BagOfWords/model_{n_bow}_BoW.pt')
        else:
            early_stopping_counter += 1

        end_time = time.time()
        print(f'Epoch [{epoch + 1}/{n_epochs}], '
              f'Loss: {mean_losses:.4f}, '
              f'F1 Score: {f1:.4f}, '
              f'IOU Score: {iou:.4f}, '
              f'Time: {round(end_time - start_time, 2)} s')

        if early_stopping_counter >= patience:
            print(f"No improvement in IOU score for {patience} consecutive epochs. Stopping early.")
            break


def get_predictions(model, dataloader):
    targets = []
    predictions = []
    model.eval()
    with torch.no_grad():
        for signals, reports in dataloader:
            signals = signals.unsqueeze(2).to(DEVICE).float()
            reports = reports.to(DEVICE).float()
            leads = torch.ones(signals.shape[0], signals.shape[1]).to(DEVICE).float() # all 12 leads

            y, p = model(signals, leads)
            targets.append(reports.cpu().numpy())
            predictions.append(p.cpu().numpy())

    targets = np.concatenate(targets, axis=0)
    predictions = np.concatenate(predictions, axis=0)
    predictions_binary = (predictions >= 0.5).astype(int)
    return targets, predictions_binary


def get_metrics(model, dataloader, average='samples'):
    y, p = get_predictions(model, dataloader)
    f1 = f1_score(y, p, average=average)
    iou = jaccard_score(y, p, average=average)
    return f1, iou


if __name__ == '__main__':
    import os
    from model import NN
    from dataset import PtbXlDataset

    os.chdir('../../')

    n_BoW = 20
    dataset = PtbXlDataset('data_ptb-xl/', 'train', n_BoW)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    val_dataset = PtbXlDataset('data_ptb-xl/', 'val', n_BoW)
    val_dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = NN(n_BoW).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    train(model, dataloader, val_dataloader, criterion, optimizer, n_epochs=500)

    # torch.save(model.state_dict(), './models/m02_ISIBrnoAIMT_BagOfWords/model_20_BoW.pt')

