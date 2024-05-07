import time
import math
import torch
import numpy as np
from torch import optim, nn
from sklearn.metrics import f1_score, jaccard_score
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import corpus_bleu
from rouge import Rouge
from nltk.translate import meteor_score


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

def print_first_n_target_prediction(dataloader, encoder, decoder, output_lang, n=10):
    count = 0
    for signals, reports in dataloader:
        encoder_outputs, encoder_hidden = encoder(signals.permute(0, 2, 1))
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden)
        predicted_indices = decoder_outputs.topk(1)[1].squeeze().detach()
        predicted_indices = predicted_indices.cpu().numpy()
        predicted_texts = [' '.join(ids_to_text(prediction, output_lang.index2word)) for prediction in predicted_indices]

        for pred, report in zip(predicted_texts, reports):
            if count < n:
                decoded_words = []
                for idx in report:
                    if idx.item() == output_lang.word2index['<eos>']:
                        break
                    decoded_words.append(output_lang.index2word[idx.item()])
                print('=', ' '.join(decoded_words))
                print('<', pred)
                print('')
                n += 1
            else:
                break
        if count < n:
            break


def one_hot_encode(sequences, num_classes, offset=3): # offset to remove '<sos>', '<eos>' and '<pad>'
    num_classes = num_classes - 3
    one_hot = np.zeros((len(sequences), num_classes), dtype=int)
    for i, sequence in enumerate(sequences):
        filtered_indices = sequence - offset
        valid_indices = filtered_indices[filtered_indices >= 0]
        one_hot[i, valid_indices] = 1
    return one_hot

def ids_to_text(ids, id_to_word):
    return [id_to_word[id] for id in ids if id > 2]

def calculate_rouge_scores(predicted_texts, reference_texts):
    try:
        rouge = Rouge()
        scores = rouge.get_scores(predicted_texts, reference_texts, avg=True)
    except ValueError:
        scores = {
            'rouge-1': {'f': 0.0, 'p': 0.0, 'r': 0.0},
            'rouge-2': {'f': 0.0, 'p': 0.0, 'r': 0.0},
            'rouge-l': {'f': 0.0, 'p': 0.0, 'r': 0.0}
        }
    return scores

def calculate_meteor_score(hypotheses, references):
    scores = [meteor_score.single_meteor_score(ref, hyp) for ref, hyp in zip(references, hypotheses)]
    avg_score = sum(scores) / len(scores)
    return avg_score

def validate_epoch(dataloader, encoder, decoder, criterion, output_lang):
    encoder.eval()
    decoder.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for input_tensor, target_tensor in dataloader:
            encoder_outputs, encoder_hidden = encoder(input_tensor.permute(0, 2, 1))
            decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)
            predicted_indices = decoder_outputs.topk(1)[1].squeeze().detach()

            loss = criterion(
                decoder_outputs.view(-1, decoder_outputs.size(-1)),
                target_tensor.view(-1)
            )
            total_loss += loss.item()

            all_predictions.append(predicted_indices.cpu().numpy())
            all_targets.append(target_tensor.cpu().numpy())

        all_predictions = np.concatenate(all_predictions)
        all_targets = np.concatenate(all_targets)

        one_hot_predictions = one_hot_encode(all_predictions, output_lang.n_words)
        one_hot_targets = one_hot_encode(all_targets, output_lang.n_words)

        f1 = f1_score(one_hot_targets, one_hot_predictions, average='macro', zero_division=0)
        jaccard = jaccard_score(one_hot_targets, one_hot_predictions, average='macro', zero_division=0)

        predicted_texts = [' '.join(ids_to_text(p, output_lang.index2word)) for p in all_predictions]
        reference_texts = [' '.join(ids_to_text(r, output_lang.index2word)) for r in all_targets]
        rouge_scores = calculate_rouge_scores(reference_texts, predicted_texts)

        predicted_texts = [' '.join(ids_to_text(prediction, output_lang.index2word)) for prediction in all_predictions]
        reference_texts = [[' '.join(ids_to_text(reference, output_lang.index2word))] for reference in all_targets]

        predicted_tokens = [ids_to_text(p, output_lang.index2word) for p in all_predictions]
        reference_tokens = [ids_to_text(r, output_lang.index2word) for r in all_targets]
        meteor_score = calculate_meteor_score(predicted_tokens, reference_tokens)

        return total_loss / len(dataloader), f1, jaccard, rouge_scores, meteor_score


def train(train_dataloader, val_dataloader, encoder, decoder, criterion, output_lang, n_epochs, learning_rate=0.001, patience=5):
    start = time.time()

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)

    best_score = 0.0
    early_stopping_counter = 0

    for epoch in range(1, n_epochs + 1):
        train_loss = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        val_loss, f1, jaccard, rouge, meteor = validate_epoch(val_dataloader, encoder, decoder, criterion, output_lang)

        print(f'{timeSince(start, epoch / n_epochs)} ({epoch} {round(epoch / n_epochs * 100, 2)}%) | '
              f'Train Loss: {round(train_loss, 4)} | Val Loss: {round(val_loss, 4)} | '
              f'Jaccard: {round(jaccard, 4)} | F1: {round(f1, 4)} | ')
        print(f'\tRouge-1 (p): {round(rouge["rouge-1"]["p"], 4)} | '
              f'Rouge-1 (r): {round(rouge["rouge-1"]["r"], 4)} | '
              f'Rouge-1 (f1): {round(rouge["rouge-1"]["f"], 4)}')
        print(f'\tRouge-2 (p): {round(rouge["rouge-2"]["p"], 4)} | '
              f'Rouge-2 (r): {round(rouge["rouge-2"]["r"], 4)} | '
              f'Rouge-2 (f1): {round(rouge["rouge-2"]["f"], 4)}')
        print(f'\tRouge-L (p): {round(rouge["rouge-l"]["p"], 4)} | '
              f'Rouge-L (r): {round(rouge["rouge-l"]["r"], 4)} | '
              f'Rouge-L (f1): {round(rouge["rouge-l"]["f"], 4)}')
        print(f'\tMETEOR: {round(meteor, 4)}')

        if meteor > best_score:
            best_score = meteor
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= patience:
            print(f"No improvement in bleu score for {patience} consecutive epochs. Stopping early.")
            break


if __name__ == '__main__':
    from dataset import get_dataloader, tensorFromSentence
    from model import NN, DecoderRNN, AttnDecoderRNN
    import os
    os.chdir('../../')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Load Train Data ...')
    _lang, dataloader = get_dataloader(file_path='./data_ptb-xl', batch_size=64, mode='debug', device=device)
    print('Load Val Data ...')
    _, val_dataloader = get_dataloader(file_path='./data_ptb-xl', batch_size=64, mode='debug', device=device, _lang=_lang)

    # Model instantiation
    hidden_size = 256

    encoder = NN(num_leads=12, hidden_size=hidden_size).to(device)
    decoder = AttnDecoderRNN(hidden_size=hidden_size, output_size=_lang.n_words, max_len=_lang.max_len).to(device)
    criterion = nn.NLLLoss()

    print('Start Training ...')
    train(dataloader, val_dataloader, encoder, decoder, criterion, _lang, 40)

    print('Test Randomly ...')
    _, test_dataloader = get_dataloader(file_path='data_ptb-xl', batch_size=64, mode='debug', device=device, _lang=_lang)

    encoder.eval()
    decoder.eval()
    print_first_n_target_prediction(test_dataloader, encoder, decoder, _lang)




