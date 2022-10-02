import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.init as init
import math


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.size = self.input_size + self.hidden_size

        # forget weight/bias
        self.W_f = Parameter(torch.Tensor(self.size, self.hidden_size))
        self.b_f = Parameter(torch.Tensor(self.hidden_size))

        # Memory weight/bias
        self.W_m = Parameter(torch.Tensor(self.size, self.hidden_size))
        self.b_m = Parameter(torch.Tensor(self.hidden_size))

        # Output weight/bias
        self.W_o = Parameter(torch.Tensor(self.size, self.hidden_size))
        self.b_o = Parameter(torch.Tensor(self.hidden_size))

        # Cell weight/bias
        self.W_c = Parameter(torch.Tensor(self.size, self.hidden_size))
        self.b_c = Parameter(torch.Tensor(self.hidden_size))

    def forward(self, input_state, hidden_state, cell_state):
        """
        :param input_state: [seq_length, batch_size, input_dim]
        :param hidden_state: [seq_length, batch_size, hidden_size]
        :param cell_state: [seq_length, batch_size, hidden_size]
        :return:
        """

        output_seq = []
        seq_length = 1
        hidden_input = torch.cat([input_state, hidden_state], 2)
        for i in range(seq_length):
            # forget gate
            forget = torch.sigmoid(torch.matmul(hidden_input, self.W_f) + self.b_f)

            # Memory gate
            memory = torch.sigmoid(torch.matmul(hidden_input, self.W_m) + self.b_m)
            C_hat = torch.tanh(torch.matmul(hidden_input, self.W_c) + self.b_c)
            print("1multiple", forget * cell_state)
            print("2multiple", memory * C_hat)
            C_t = forget * cell_state + memory * C_hat
            print("C", C_t)

            # Output gate
            Output = torch.sigmoid(torch.matmul(hidden_input, self.W_o) + self.b_o)
            hidden = Output * torch.tanh(C_t)

            output_seq.append(hidden)

        output_seq = torch.cat(output_seq, dim=0)

        return output_seq, hidden, C_t


def reset_weigths(model):
    """reset weights
    """
    for weight in model.parameters():
        init.constant_(weight, 0.5)


inputs = torch.ones(1, 1, 10)
# hiddens = torch.ones(1, 1, 20)
# cells = torch.ones(1, 1,  20)
cells = torch.randn(1, 1, 20)
hiddens = torch.randn(1, 1, 20)
myLstm = LSTM(10, 20)
reset_weigths(myLstm)
Output, hidden, C_t = myLstm(inputs, hiddens, cells)

# lstm = nn.LSTM(10, 20)
# reset_weigths(lstm)
# Output_o, (hidden_o, C_t_o) = lstm(inputs, (hiddens, cells))
# print(hidden.shape)
# print(C_t.shape)
# print(Output.shape)
#
# print(hidden)
# print(C_t)
# print(Output)
# print(hidden_o)
# print(C_t_o)
# print(Output_o)

