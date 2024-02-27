import os
import time
import numpy as np
import torch
import tqdm
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score


DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def train(model, dataloader, val_dataloader, criterion, optimizer, n_epochs=10):
    best_score = 0.0

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
            sparsity_loss = torch.mean(-4*p*(p-1))
            J = loss + sparsity_loss
            J.backward()
            optimizer.step()

            losses.append(J.item())

        mean_losses = np.mean(losses)
        val_score = valid_part(model, val_dataloader)

        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), './models/02_ISIBrnoAIMT_BagOfWords/model.pt')

        end_time = time.time()
        print(f'Epoch [{epoch + 1}/{n_epochs}], '
              f'Loss: {mean_losses:.4f}, '
              f'Validation Score: {val_score:.4f}, '
              f'Time: {round(end_time - start_time, 2)} s')


def valid_part(model, dataloader):
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

    f1 = f1_score(targets, predictions_binary, average='micro')
    return f1


if __name__ == '__main__':
    import os
    from model import NN
    from dataset import PtbXlDataset

    os.chdir('../../')

    n_BoW = 50
    dataset = PtbXlDataset('data_ptb-xl/', 'train', n_BoW)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    val_dataset = PtbXlDataset('data_ptb-xl/', 'val', n_BoW)
    val_dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = NN(n_BoW).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    train(model, dataloader, val_dataloader, criterion, optimizer, n_epochs=500)

    # torch.save(model.state_dict(), './models/02_ISIBrnoAIMT_BagOfWords/model.pt')

