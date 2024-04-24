import time
import math
import torch
from torch import optim, nn
from sklearn.metrics import f1_score
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import corpus_bleu

import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from dataset import get_dataloader, tensorFromSentence
from model import EncoderRNN, DecoderRNN, AttnDecoderRNN



def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def train_epoch(dataloader, encoder, decoder, encoder_optimizer,
          decoder_optimizer, criterion):
    encoder.train()
    decoder.train()
    total_loss = 0
    for data in dataloader:
        input_tensor, target_tensor = data

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_outputs, encoder_hidden = encoder(input_tensor.permute(0, 2, 1))
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)

        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            target_tensor.view(-1)
        )
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def evaluate(encoder, decoder, input_tensor, output_lang):
    with torch.no_grad():
        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, decoder_hidden, decoder_attn = decoder(encoder_outputs, encoder_hidden)

        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        decoded_words = []
        for idx in decoded_ids:
            if idx.item() == output_lang.word2index['<eos>']:
                decoded_words.append('<eos>')
                break
            decoded_words.append(output_lang.index2word[idx.item()])
    return decoded_words, decoder_attn

def evaluateRandomly(encoder, decoder, output_lang, n=1):
    for sig, report in dataloader:
        decoded_words = []
        for idx in report[0]:
            if idx.item() == output_lang.word2index['<eos>']:
                decoded_words.append('<eos>')
                break
            decoded_words.append(output_lang.index2word[idx.item()])
        print('=', ' '.join(decoded_words))
        sig = sig.permute(0, 2, 1)[0].unsqueeze(0)
        output_words, _ = evaluate(encoder, decoder, sig, output_lang)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')

def evaluate_batch(decoder_outputs, output_lang):
    with torch.no_grad():
        # decoder_outputs shape: [batch_size, seq_len, output_vocab_size]
        # We apply topk to get the most likely token index at each position in the sequence for each batch
        _, topi = decoder_outputs.topk(1)
        batch_decoded_words = []

        # Iterate over each sequence in the batch
        for b in range(topi.size(0)):
            decoded_ids = topi[b].squeeze().tolist()  # Get predicted sequence for this batch item
            decoded_words = []

            for idx in decoded_ids:
                if idx == output_lang.word2index['<eos>']:
                    decoded_words.append('<eos>')
                    break
                else:
                    decoded_words.append(output_lang.index2word[idx])

            batch_decoded_words.append(decoded_words)

    return batch_decoded_words


def validate_epoch(dataloader, encoder, decoder, criterion, output_lang):
    encoder.eval()
    decoder.eval()
    total_loss = 0
    all_true_words = []
    all_pred_words = []

    with torch.no_grad():
        for data in dataloader:
            input_tensor, target_tensor = data

            encoder_outputs, encoder_hidden = encoder(input_tensor.permute(0, 2, 1))
            decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)

            loss = criterion(
                decoder_outputs.view(-1, decoder_outputs.size(-1)),
                target_tensor.view(-1)
            )
            total_loss += loss.item()

            true_words_batch = [[output_lang.index2word[idx.item()] for idx in seq] for seq in target_tensor]
            predicted_words_batch = evaluate_batch(decoder_outputs, output_lang)

            all_true_words.extend(true_words_batch)
            all_pred_words.extend(predicted_words_batch)

    # flat_true_words = [word for sublist in all_true_words for word in sublist]
    # flat_pred_words = [word for sublist in all_pred_words for word in sublist]

    f1 = f1_score(all_true_words, all_pred_words, average='weighted', labels=list(output_lang.word2index.values()))

    # Calculate BLEU score
    bleu_scores = [sentence_bleu([ref], pred) for ref, pred in zip(all_true_words, all_pred_words)]
    bleu = sum(bleu_scores) / len(bleu_scores)

    return total_loss / len(dataloader), f1, bleu


def train(train_dataloader, val_dataloader, encoder, decoder, output_lang, n_epochs, learning_rate=0.001,
               print_every=100):
    start = time.time()
    plot_losses = []
    train_loss_total = 0
    val_loss_total = 0

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    for epoch in range(1, n_epochs + 1):
        train_loss = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        val_loss, f1, bleu = validate_epoch(val_dataloader, encoder, decoder, criterion, output_lang)
        # val_loss = validate_epoch(val_dataloader, encoder, decoder, criterion, output_lang)
        train_loss_total += train_loss
        val_loss_total += val_loss

        if epoch % print_every == 0:
            train_loss_avg = train_loss_total / print_every
            val_loss_avg = val_loss_total / print_every
            train_loss_total = 0
            val_loss_total = 0
            print(f'{timeSince(start, epoch / n_epochs)} ({epoch} {epoch / n_epochs * 100}%) | '
                  f'Train Loss: {round(train_loss_avg, 4)} | Val Loss: {round(val_loss_avg, 4)} | '
                  f'BLEU: {round(bleu, 4)} | F1: {round(f1, 4)}')


if __name__ == '__main__':
    from dataset import get_dataloader, tensorFromSentence
    from model import EncoderRNN, DecoderRNN, AttnDecoderRNN
    import os
    os.chdir('../../')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Load Train Data ...')
    _lang, dataloader = get_dataloader(file_path='data_ptb-xl', batch_size=64, mode='debug', device=device)
    print('Load Val Data ...')
    _, val_dataloader = get_dataloader(file_path='data_ptb-xl', batch_size=64, mode='debug', device=device, _lang=_lang)

    # Model instantiation
    hidden_size = 128

    encoder = EncoderRNN(num_leads=12, hidden_size=hidden_size).to(device)
    # EncoderRNN(input_size=1000, hidden_size=hidden_size).to(device)
    decoder = AttnDecoderRNN(hidden_size=hidden_size, output_size=_lang.n_words, max_len=_lang.max_len).to(device)
    print('Start Training ...')
    # Start training
    train(dataloader, val_dataloader, encoder, decoder, _lang, 40, print_every=10)

    print('Eval Randomly ...')
    encoder.eval()
    decoder.eval()
    evaluateRandomly(encoder, decoder, _lang)




