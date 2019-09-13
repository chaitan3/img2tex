import torch
from torch.nn.functional import tanh, softmax, log_softmax

conv_feature_size = 64
#rnn_encoder_hidden_size = 256
rnn_encoder_hidden_size = 32
rnn_decoder_hidden_size = 512
tex_token_size = 556
tex_embedding_size = 80
rnn_max_steps = 150
SOS_token = tex_token_size - 3 
EOS_token = tex_token_size - 1

device = torch.device('cuda:0')

def tinfo(a):
    a = torch.abs(a)
    return (a.min().item(), a.max().item())

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
        self.hidden_size = rnn_decoder_hidden_size
        self.embedding_size = tex_embedding_size
        self.context_size = 2*rnn_encoder_hidden_size
        self.token_size = tex_token_size
        self.score_size = self.hidden_size 
        self.out_size = self.hidden_size 
        self.max_steps = rnn_max_steps

        self.embedding_layer = torch.nn.Embedding(self.token_size, self.embedding_size)
        #self.rnn = torch.nn.LSTM(self.embedding_size + self.out_size, self.hidden_size)
        self.rnn = torch.nn.GRU(self.embedding_size + self.out_size, self.hidden_size)
        self.score_matrix_layer1 = torch.nn.Linear(self.hidden_size, self.score_size, bias=False)
        self.score_matrix_layer2 = torch.nn.Linear(self.context_size, self.score_size, bias=False)
        self.score_vector_layer = torch.nn.Linear(self.score_size, 1, bias=False)
        self.context_layer = torch.nn.Linear(self.hidden_size + self.context_size, self.out_size, bias=False)
        self.out_layer = torch.nn.Linear(self.out_size, self.token_size, bias=False)

        
    def forward(self, rnn_enc, decoded_outputs=None):
        N, rnn_enc_size = rnn_enc.shape[:2]
        conv_size = rnn_enc.shape[2]*rnn_enc.shape[3]
        rnn_enc = rnn_enc.reshape(N, rnn_enc_size, -1).permute(0, 2, 1)
        # define initial states
        hidden = torch.zeros(1, N, self.hidden_size, device=device)
        cell = torch.zeros(1, N, self.hidden_size, device=device)
        out = torch.zeros(N, self.out_size, device=device)

        token = SOS_token*torch.ones(N, dtype=torch.long, device=device)
        p_tokens = []
        print(rnn_enc.shape, decoded_outputs.shape)

        for step in range(0, self.max_steps):
            
            # embedding layer
            inp = self.embedding_layer(token).reshape(1, N, -1)
            inp = torch.cat((inp, out.reshape(1, N, -1)), dim=2)
            #print('inp', tinfo(inp))

            # rnn step
            #_, (hidden, cell) = self.rnn(inp, (hidden, cell))
            _, hidden = self.rnn(inp, hidden)
            #print('hidden', tinfo(hidden))

            # attention mechanism from previous step
            hidden_cast = hidden.permute(1, 0, 2)
            score = self.score_vector_layer(tanh(self.score_matrix_layer1(hidden_cast) + self.score_matrix_layer2(rnn_enc)))
            attention = softmax(score, dim=1)
            context = (attention*rnn_enc).sum(dim=1)
            #print('context', tinfo(context))

            # compute outputs
            hidden_cast = hidden_cast.reshape(N, -1)
            out = tanh(self.context_layer(torch.cat((hidden_cast, context), dim=1)))
            p_token = softmax(self.out_layer(out), dim=1)

            p_tokens.append(p_token.reshape(N, -1, 1))
            print('p_tokens', tinfo(p_token))
            #import pdb;pdb.set_trace()

            if decoded_outputs is not None:
                token = decoded_outputs[:, step]
                if step + 1 == decoded_outputs.shape[1]:
                    break
            else:
                assert N == 1
                token = p_token.topk(1)[1].squeeze().detach()
                if token == EOS_token:
                    break
                        
        #exit(1)
        return torch.cat(p_tokens, dim=2)

class Img2Tex(torch.nn.Module):
    def __init__(self):
        super(Img2Tex, self).__init__()
        self.cnn_encoder = ConvolutionalEncoder()
        #self.rnn_encoder = RNNEncoder()
        self.rnn_decoder = RNNDecoder()

    def forward(self, x, y=None):
        cnn_enc = self.cnn_encoder(x)
        #rnn_enc = self.rnn_encoder(cnn_enc)
        rnn_enc = cnn_enc
        y_pred = self.rnn_decoder(rnn_enc, y)
        return y_pred
