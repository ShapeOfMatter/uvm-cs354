import torch

from state import ModelSettings

ALPHABET_SIZE = 256

class RNNModel(torch.nn.Module):
    def __init__(self, name: str, settings: ModelSettings):
        super().__init__()
        self.name = name
        self.embed = torch.nn.Embedding(num_embeddings=ALPHABET_SIZE,
                                        embedding_dim=settings.embedding_width)
        self.rnn = torch.nn.RNN(input_size=settings.embedding_width,
                                hidden_size=settings.hidden_width,
                                num_layers=settings.rnn_height,
                                nonlinearity=settings.nonlinear_function,
                                dropout=settings.dropout)
        self.judge = torch.nn.Linear(settings.hidden_width, ALPHABET_SIZE)

    def forward(self, input_i, state_i_minus1):
        output_i, state_i = self.rnn(self.embed(input_i), state_i_minus1)
        return self.judge(output_i), state_i

    def state_zero(self, length: int):
        return torch.zeros(self.rnn.num_layers, length, self.rnn.hidden_size)
               # torch.zeros(self.rnn.num_layers, length, self.rnn.hidden_size))

    def detach_state(self, state):
        return state.detach()

