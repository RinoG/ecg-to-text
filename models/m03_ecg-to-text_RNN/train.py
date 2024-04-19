import time
import math
import torch

from dataset import get_dataloader, tensorFromSentence
from model import EncoderRNN, DecoderRNN, AttnDecoderRNN
from torch import optim, nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

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


def train(train_dataloader, encoder, decoder, n_epochs, learning_rate=0.001,
               print_every=100):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    for epoch in range(1, n_epochs + 1):
        loss = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, epoch / n_epochs),
                                        epoch, epoch / n_epochs * 100, print_loss_avg))

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


if __name__ == '__main__':
    from dataset import get_dataloader, tensorFromSentence
    from model import EncoderRNN, DecoderRNN, AttnDecoderRNN
    import os
    os.chdir('../../')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Load Data ...')
    _lang, dataloader = get_dataloader(file_path='data_ptb-xl', batch_size=32, mode='train', device=device)

    # Model instantiation
    hidden_size = 128

    encoder = EncoderRNN(input_size=5000, hidden_size=hidden_size).to(device)
    decoder = AttnDecoderRNN(hidden_size=hidden_size, output_size=_lang.n_words, max_len=_lang.max_len).to(device)
    print('Start Training ...')
    # Start training
    train(dataloader, encoder, decoder, 50, print_every=5)

    print('Load Eval data ...')
    _lang, dataloader = get_dataloader(file_path='data_ptb-xl', batch_size=64, mode='val', device=device)
    print('Eval Randomly ...')
    encoder.eval()
    decoder.eval()
    evaluateRandomly(encoder, decoder, _lang)




