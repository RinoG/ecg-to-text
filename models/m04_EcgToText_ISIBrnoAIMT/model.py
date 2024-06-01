import torch
import torch.nn as nn
import torch.nn.functional as F

SOS_token = 1
MAX_LENGTH = 50
teacher_forcing_ratio = 0.2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MyResidualBlock(nn.Module):
    def __init__(self, down_sample, hidden_size=256):
        super(MyResidualBlock, self).__init__()
        self.down_sample = down_sample
        self.stride = 2 if self.down_sample else 1
        K = 9
        P = (K-1) // 2
        self.conv1 = nn.Conv2d(in_channels=hidden_size,
                               out_channels=hidden_size,
                               kernel_size=(1, K),
                               stride=(1, self.stride),
                               padding=(0, P),
                               bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_size)

        self.conv2 = nn.Conv2d(in_channels=hidden_size,
                               out_channels=hidden_size,
                               kernel_size=(1, K),
                               padding=(0, P),
                               bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_size)

        if self.down_sample:
            self.idfunc_0 = nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2))
            self.idfunc_1 = nn.Conv2d(in_channels=hidden_size,
                                      out_channels=hidden_size,
                                      kernel_size=(1, 1),
                                      bias=False)
            self.idfunc_1_bn = nn.BatchNorm2d(hidden_size)

    def forward(self, x):
        identity = x
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        if self.down_sample:
            identity = self.idfunc_0(identity)
            identity = self.idfunc_1(identity)
            identity = self.idfunc_1_bn(identity)

        if identity.size(2) != x.size(2) or identity.size(3) != x.size(3):
            diff = x.size(3) - identity.size(3)
            if diff > 0:  # x is larger
                identity = F.pad(identity, (0, diff), "constant", 0)
            else:  # identity is larger, need to crop or pool (example uses cropping)
                identity = identity[:, :, :, :x.size(3)]

        x = x+identity
        return x


class NN(nn.Module):
    def __init__(self, num_leads, hidden_size):
        super(NN, self).__init__()
        self.conv = nn.Conv2d(in_channels=num_leads,
                              out_channels=hidden_size,
                              kernel_size=(1, 15),
                              padding=(0, 7),
                              stride=(1, 2),
                              bias=False)
        self.bn = nn.BatchNorm2d(hidden_size)
        self.rb_0 = MyResidualBlock(down_sample=True, hidden_size=hidden_size)
        self.rb_1 = MyResidualBlock(down_sample=True, hidden_size=hidden_size)
        self.rb_2 = MyResidualBlock(down_sample=True, hidden_size=hidden_size)
        self.rb_3 = MyResidualBlock(down_sample=True, hidden_size=hidden_size)
        self.rb_4 = MyResidualBlock(down_sample=True, hidden_size=hidden_size)

        self.mha = nn.MultiheadAttention(hidden_size, 8)
        self.hidden_transform = nn.Linear(16, hidden_size)

    def forward(self, x):
        x = x.unsqueeze(2)
        x = F.leaky_relu(self.bn(self.conv(x)))

        x = self.rb_0(x)
        x = self.rb_1(x)
        x = self.rb_2(x)
        x = self.rb_3(x)
        x = self.rb_4(x)

        x = F.dropout(x, p=0.3, training=self.training)

        x = x.squeeze(2).permute(2,0,1)
        x, hidden = self.mha(x, x, x)
        hidden = self.hidden_transform(hidden.mean(dim=1))
        hidden = hidden.unsqueeze(0)
        x = x.permute(1, 0, 2)
        return x, hidden



class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, max_len):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.max_len = max_len

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []

        for i in range(self.max_len):
            decoder_output, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs, decoder_hidden, None # We return `None` for consistency in the training loop

    def forward_step(self, input, hidden):
        output = self.embedding(input)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        return output, hidden


class BahdanauAttention(nn.Module):
    def __init__(self, encoder_hidden_size, decoder_hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(decoder_hidden_size, decoder_hidden_size)
        self.Ua = nn.Linear(encoder_hidden_size, decoder_hidden_size)
        self.Va = nn.Linear(decoder_hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, encoder_hidden_size, output_size, max_len, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.max_len = max_len
        self.hidden_transform = nn.Linear(encoder_hidden_size, hidden_size)
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = BahdanauAttention(encoder_hidden_size, hidden_size)
        self.gru = nn.GRU(encoder_hidden_size + hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
        decoder_hidden = self.hidden_transform(encoder_hidden)
        decoder_outputs = []
        attentions = []

        for i in range(self.max_len):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions

    def forward_step(self, input, hidden, encoder_outputs):
        embedded = self.dropout(self.embedding(input))

        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=2)

        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)

        return output, hidden, attn_weights


class TransformerDecoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout_p=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout_p)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.ReLU(),
            nn.Linear(4 * hidden_size, hidden_size)
        )
        self.layernorm1 = nn.LayerNorm(hidden_size)
        self.layernorm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        tgt2 = self.layernorm1(tgt + self.dropout(self.multihead_attn(tgt, tgt, tgt, attn_mask=tgt_mask)[0]))
        tgt3 = self.layernorm2(tgt2 + self.dropout(self.feed_forward(tgt2)))
        return tgt3


class TransformerDecoder(nn.Module):
    def __init__(self, output_size, hidden_size, num_layers, num_heads, dropout_p=0.1):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.pos_encoder = nn.Parameter(torch.randn(1, 64, hidden_size))
        self.layers = nn.ModuleList([TransformerDecoderLayer(hidden_size, num_heads, dropout_p) for _ in range(num_layers)])
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, src, memory, src_mask=None, memory_mask=None):
        # src.shape = torch.Size([64, 16, 512])
        # src = src.long()
        # src.shape = torch.Size([64, 16, 512])
        # src = self.embedding(src)
        # src.shape = torch.Size([64, 16, 512, 512])
        pos_encoding = self.pos_encoder[:, :src.size(1), :]
        # self.pos_encoder.shape = torch.Size([1, 64, 512])
        # pos_encoding.shape = torch.Size([1, 16, 512])
        src += pos_encoding
        for layer in self.layers:
            src = layer(src, memory, tgt_mask=src_mask, memory_mask=memory_mask)
        output = self.linear(src)
        return F.log_softmax(output, dim=-1)


if __name__ == '__main__':
    import torch

    # hidden_size = 128
    # hidden_size = 256
    # hidden_size = 512
    # hidden_size = 1024
    encoder_hidden_size = 512
    decoder_hidden_size = 256

    x = torch.randn(64, 12, 1000).to(device)
    target = torch.randint(0, 2788, (64, 64)).to(device)

    encoder = NN(num_leads=12, hidden_size=encoder_hidden_size).to(device)
    # decoder = AttnDecoderRNN(hidden_size=decoder_hidden_size,
    #                          encoder_hidden_size=encoder_hidden_size,
    #                          output_size=2788,
    #                          max_len=64).to(device)
    decoder = TransformerDecoder(output_size=2788, hidden_size=512, num_layers=6, num_heads=8).to(device)

    n_encoder = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    n_decoder = sum(p.numel() for p in decoder.parameters() if p.requires_grad)

    print(f"Encoder: {n_encoder}; Decoder: {n_decoder}")

    encoder_outputs, encoder_hidden = encoder(x)
    decoder_outputs, decoder_hidden, attn_weights = decoder(encoder_outputs, encoder_hidden, target)
    print()
