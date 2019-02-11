import torch
from torch.nn.functional import tanh, softmax, log_softmax

conv_feature_size = 64
rnn_decoder_hidden_size = 512
rnn_encoder_hidden_size = 256
tex_token_size = 554
tex_embedding_size = 80
rnn_max_steps = 150

device = torch.device('cuda:0')

class ConvolutionalEncoder(torch.nn.Module):
    def __init__(self):
        super(ConvolutionalEncoder, self).__init__()
        conv1 = torch.nn.Conv2d(1, 512, 3)
        bn1 = torch.nn.BatchNorm2d(512)
        conv2 = torch.nn.Conv2d(512, 512, 3, padding=1)
        bn2 = torch.nn.BatchNorm2d(512)
        pool1 = torch.nn.MaxPool2d((1,2), stride=(1,2))

        conv3 = torch.nn.Conv2d(512, 256, 3, padding=1)
        pool2 = torch.nn.MaxPool2d((2,1), stride=(2,1))
        conv4 = torch.nn.Conv2d(256, 256, 3, padding=1)
        bn3 = torch.nn.BatchNorm2d(256)

        conv5 = torch.nn.Conv2d(256, 128, 3, padding=1)
        pool3 = torch.nn.MaxPool2d((2,2), stride=(2,2))
        conv6 = torch.nn.Conv2d(128, conv_feature_size, 3, padding=1)
        pool4 = torch.nn.MaxPool2d((2,2), stride=(2,2))
        #pool4 = torch.nn.MaxPool2d((2,2), stride=(2,2), padding=2)
        
        self.model = torch.nn.Sequential(
                conv1, bn1, conv2, bn2, pool1,
                conv3, pool2, conv4, bn3,
                conv5, pool3, conv6, pool4
                )

    def forward(self, x):
        return self.model(x)

class RNNEncoder(torch.nn.Module):
    def __init__(self):
        super(RNNEncoder, self).__init__()
        self.hidden_size = rnn_encoder_hidden_size
        self.rnn = torch.nn.LSTM(conv_feature_size, self.hidden_size, bidirectional=True)

    def forward(self, conv_feats):
        #return conv_feats
        N = conv_feats.shape[0]
        outs = []
        num_rows, num_cols = conv_feats.shape[2:]
        for i in range(0, num_rows):
            h0 = torch.zeros(2, N, self.hidden_size, device=device)
            c0 = torch.zeros(2, N, self.hidden_size, device=device)
            seq = conv_feats[:,:,i,:].permute(2, 0, 1)
            out_seq = self.rnn(seq, (h0, c0))[0].reshape(num_cols, N, 1, 2*self.hidden_size).permute(1, 3, 2, 0)
            outs.append(out_seq)
        return torch.cat(outs, dim=2)

class RNNDecoder(torch.nn.Module):
    def __init__(self):
        super(RNNDecoder, self).__init__()
        #self.feature_size = conv_feature_size
        self.feature_size = 2*rnn_encoder_hidden_size
        self.hidden_size = rnn_decoder_hidden_size
        self.embedding_size = tex_embedding_size
        self.token_size = tex_token_size
        self.out_size = self.hidden_size + self.feature_size
        self.max_steps = rnn_max_steps

        self.rnn = torch.nn.LSTM(self.embedding_size, self.hidden_size)
        self.context_layer = torch.nn.Linear(self.out_size, self.embedding_size)
        self.out_layer = torch.nn.Linear(self.embedding_size, self.token_size)
        self.score_matrix_layer = torch.nn.Linear(self.out_size, self.hidden_size)
        self.score_vector_layer = torch.nn.Linear(self.hidden_size, 1)

        self.embedding_layer = torch.nn.Embedding(self.token_size, self.embedding_size)
        
    def forward(self, rnn_enc):
        N, rnn_enc_size = rnn_enc.shape[:2]
        conv_size = rnn_enc.shape[2]*rnn_enc.shape[3]
        rnn_enc = rnn_enc.reshape(N, rnn_enc_size, -1).permute(0, 2, 1)
        # define initial states
        hidden = torch.zeros(1, N, self.hidden_size, device=device)
        cell = torch.zeros(1, N, self.hidden_size, device=device)

        output = []
        for step in range(0, self.max_steps):
            # embedding layer
            #inp = out.reshape(1, N, -1)
            inp = self.embedding_layer(output[-1]).reshape(1, N, -1)

            # rnn step
            #inp = torch.cat((token, out), dim=1).reshape(1, N, -1)
            _, (hidden, cell) = self.rnn(inp, (hidden, cell))

            # attention mechanism
            hidden_cast = hidden.reshape(N, 1, -1).expand(N, conv_size, -1)
            score = self.score_vector_layer(tanh(self.score_matrix_layer(torch.cat((hidden_cast, rnn_enc), dim=2))))
            attention = softmax(score, dim=1).reshape(N, -1, 1)
            context = (attention*rnn_enc).sum(dim=1)

            # compute outputs
            out = tanh(self.context_layer(torch.cat((hidden.reshape(N, -1), context), dim=1)))
            token = log_softmax(self.out_layer(out), dim=1)
            output.append(token.reshape(N, -1, 1))
                        
        return torch.cat(output, dim=2)

class Img2Tex(torch.nn.Module):
    def __init__(self):
        super(Img2Tex, self).__init__()
        self.cnn_encoder = ConvolutionalEncoder()
        self.rnn_encoder = RNNEncoder()
        self.rnn_decoder = RNNDecoder()

    def forward(self, x):
        cnn_enc = self.cnn_encoder(x)
        rnn_enc = self.rnn_encoder(cnn_enc)
        y_pred = self.rnn_decoder(rnn_enc)
        return y_pred
