import torch

ALPHABET_SIZE = 256
NONLINEAR_FUNCTION = 'relu'
DROPOUT = 0.1

class RNNModel(torch.nn.Module):
    def __init__(self,
                 embedding_width: int = 64,
                 hidden_width: int = 300,
                 rnn_height: int = 5):
        super().__init__()
        self.embed = torch.nn.Embedding(num_embeddings=ALPHABET_SIZE,
                                        embedding_dim=embedding_width)
        self.rnn = torch.nn.RNN(input_size=embedding_width,
                                hidden_size=hidden_width,
                                num_layers=rnn_height,
                                nonlinearity=NONLINEAR_FUNCTION,
                                dropout=DROPOUT)
        self.judge = torch.nn.Linear(hidden_width, ALPHABET_SIZE)

    def forward(self, input_i, state_i_minus1):
        output_i, state_i = self.rnn(self.embed(input_i),
                                     state_i_minus1)
        return self.judge(output_i), state_i

    def state_zero(self, length):
        return (torch.zeros(self.rnn.num_layers, length, self.rnn.hidden_size),
                torch.zeros(self.rnn.num_layers, length, self.rnn.hidden_size))

