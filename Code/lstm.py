import torch
import torch.nn as nn

class CustomLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CustomLSTMCell, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size

        # Concatenated weight matrices for all gates: f, i, o, c̃
        self.W = nn.Linear(input_size + hidden_size, 4 * hidden_size)

    def forward(self, x_t, h_prev, c_prev):
        combined = torch.cat((h_prev, x_t), dim=1)  # shape: (batch, input+hidden)
        gates = self.W(combined)  # shape: (batch, 4*hidden)

        f_t, i_t, o_t, c_hat_t = torch.chunk(gates, 4, dim=1)

        f_t = torch.sigmoid(f_t)
        i_t = torch.sigmoid(i_t)
        o_t = torch.sigmoid(o_t)
        c_hat_t = torch.tanh(c_hat_t)

        c_t = f_t * c_prev + i_t * c_hat_t # this is the long term memory
        h_t = o_t * torch.tanh(c_t) # this is the final output

        return h_t, c_t


class CustomLSTMForecast(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, output_size=1, seq_len=10):
        super(CustomLSTMForecast, self).__init__()
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.lstm_cell = CustomLSTMCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        h_t = torch.zeros(batch_size, self.hidden_size, device=x.device) # init the long and short term memory
        c_t = torch.zeros(batch_size, self.hidden_size, device=x.device)

        for t in range(self.seq_len):
            x_t = x[:, t, :]  # shape: (batch, input_size)
            h_t, c_t = self.lstm_cell(x_t, h_t, c_t) # pass the inputs and memory to the sequence of cells

        out = self.fc(h_t)  # use last hidden state
        return out