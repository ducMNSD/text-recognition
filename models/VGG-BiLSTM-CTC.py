# -*- coding: utf-8 -*-
import torch.nn as nn
from torchsummary import summary


class BidirectionalLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True)
        self.embedding = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        """
        input : visual feature [seq_len x batch_size x input_size]
        output : contextual feature [seq_len x batch_size x output_size]
        """
        
        recurrent, _ = self.rnn(input)
        seq_len, batch, hidden_size = recurrent.size()
        t_rec = recurrent.view(seq_len*batch, hidden_size)

        output = self.embedding(t_rec)  # [seq_len * batch, output_size]
        output = output.view(seq_len, batch, -1)

        return output


class CRNN(nn.Module):

    def __init__(self, imgH, channels, nclass, hidden_state, n_rnn=2, leakyRelu=False):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 2]                  # kernel size
        ps = [1, 1, 1, 1, 1, 1, 0]                  # stride size
        ss = [1, 1, 1, 1, 1, 1, 1]                  # padding size
        nm = [64, 128, 256, 256, 512, 512, 512]     # maps

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            num_input = channels if i == 0 else nm[i - 1]
            num_output = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(num_input, num_output, ks[i], ss[i], ps[i]))
            
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(num_output))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)                                                 # 64x32x100
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x50
        convRelu(1)                                                 # 128x16x50
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x25
        convRelu(2, True)                                           # 256x8x25
        convRelu(3)                                                 # 256x8x25
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))        # 256x4x26
        convRelu(4, True)                                           # 512x4x26
        convRelu(5)                                                 # 512x4x26
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))        # 512x2x27
        convRelu(6, True)                                           # 512x1x26

        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, hidden_state, hidden_state),
            BidirectionalLSTM(hidden_state, hidden_state, nclass))

    def forward(self, input):
        # feature extraction
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        # sequence modelings
        output = self.rnn(conv)

        return output
    
if __name__ == "__main__":
    model = CRNN(32, 1, 66, 256)
    summary(model, (1, 32, 100))