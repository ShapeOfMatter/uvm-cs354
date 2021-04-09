from dataclasses import dataclass, replace
from dataclasses_json import dataclass_json

def read_whole_file(filename: str) -> str:
    with open(filename, 'r') as f:
        return f.read()

def write_whole_file(filename: str, contents: str) -> None:
    with open(filename, 'w') as f:
        f.write(contents)

class _Settings:
    @staticmethod
    def from_json(j: str):
        raise Exception("Not implemented.")
    
    def to_json(self, *, indent: int=0):
        raise Exception("Not implemented.")

    @classmethod
    def load(cls, filename: str) -> 'Settings':
        return cls.from_json(read_whole_file(filename))

    def save(self, filename: str) -> None:
        write_whole_file(filename, self.to_json(indent=2))


@dataclass_json
@dataclass(frozen=True)
class Settings(_Settings):
    name: str
    log_file: str
    training_source: str
    testing_source: str
    training_settings: 'TrainingSettings'
    model_settings: 'ModelSettings'
    snippet_length: int
    total_lifetime: int

@dataclass_json
@dataclass(frozen=True)
class ModelSettings(_Settings):
    embedding_width: int = 64
    hidden_width: int = 300
    rnn_height: int = 5

@dataclass_json
@dataclass(frozen=True)
class TrainingSettings(_Settings):
    batch_size: int
    learning_rate: float = 1e-3
    beta_1: float = 0.9
    beta_2: float = 0.999
    epsilon: float = 1e-8
    weight_decay: float = 0

