import os
import time
import logging
import numpy as np
import pandas as pd
import torch
import tqdm
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, jaccard_score


DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
logger = logging.getLogger(__name__)

class challengeloss(nn.Module):
    def __init__(self, train):
        super(challengeloss,self).__init__()

        n_labels = train.shape[1]
        co_occurrence_matrix = np.zeros((n_labels, n_labels))

        for i in range(n_labels):
            for j in range(n_labels):
                # Count how often label_i and label_j both have value 1 in the same row
                co_occurrence_matrix[i, j] = ((train[:, i] == 1) & (train[:, j] == 1)).sum()

        max_co_occurrence = co_occurrence_matrix.max()
        weights_matrix = 1 - (co_occurrence_matrix / (max_co_occurrence + 1))
        np.fill_diagonal(weights_matrix, 1.0)

        self.weights = torch.from_numpy(weights_matrix).float().to(DEVICE).requires_grad_(False)
        self.I = torch.ones((n_labels, n_labels)).float().to(DEVICE).requires_grad_(False)


    def forward(self, L, P):
        L = L.float()
        N = L + P - L * P
        N = torch.mm(N, self.I) + 1e-6
        C = torch.mm(L.T, P / N)
        C = torch.sum(self.weights * C)
        return C
def get_df(dataloader):
    labels_list = []

    for _, labels in dataloader:
        labels_list.append(labels.numpy())

    return np.concatenate(labels_list, axis=0)

def train(model, dataloader, val_dataloader, loss, optimizer, scheduler, n_epochs=50, patience=10):
    logger.info("Start training")
    best_score = 0.0
    early_stopping_counter = 0
    chloss = challengeloss(get_df(dataloader))

    for epoch in range(n_epochs):
        start_time = time.time()
        model.train(True)
        losses = []
        for signals, reports in dataloader:
            optimizer.zero_grad()

            signals = signals.unsqueeze(2).to(DEVICE).float()
            reports = reports.to(DEVICE).float()
            leads = torch.ones(signals.shape[0], signals.shape[1]).to(DEVICE).float()

            y, p = model(signals, leads)

            N = loss(y, reports)
            Q = torch.mean(-4 * p * (p - 1))
            M = chloss(reports, p)
            J = N + Q - M
            J.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

            optimizer.step()

            losses.append(J.item())

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

        # scheduler.step()


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

    loss_weight = (len(dataset) - dataset.summary(output='numpy')) / dataset.summary(output='numpy')

    model = NN(n_BoW).to(DEVICE)
    # criterion = nn.BCEWithLogitsLoss()
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(loss_weight).to(DEVICE))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    train(model, dataloader, val_dataloader, bce_loss, optimizer, scheduler, n_epochs=50)

    # torch.save(model.state_dict(), './models/m02_BagOfWords_ISIBrnoAIMT/model_20_BoW.pt')

