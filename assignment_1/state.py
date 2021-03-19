from contextlib import contextmanager
from dataclasses import dataclass
from dataclasses_json import dataclass_json
from filelock import FileLock
from itertools import count, repeat
from pprint import pprint
import random
from subprocess import run
from time import time
from typing import Dict, Optional, Tuple

@dataclass_json
@dataclass(fronzen=True)
class TrainingSettings:
    name: str
    learning_rate: float
    momentum: float
    weight_decay: float = 0.0005

@dataclass_json
@dataclass(frozen=True)
class Settings:
    lock_file: str
    lock_timeout: int
    state_file: str
    log_file: str
    training_source_dir: str
    testing_source_dir: str
    model_name_format: str
    num_models: int
    pretrained_features: bool
    pretrained_classifier: bool
    trainable_features: Tuple[bool...]
    trainable_classifier: Tuple[bool...]
    base_trainer_profiles: Tuple[TrainingSettings...]
    max_batch_size: int
    max_seconds: int
    total_lifetime: int
    eon_lifetime: int

def read_whole_file(filename: str) -> str:
    with open(filename, 'r') as f:
        return f.read()

def write_whole_file(filename: str, contents: str) -> None:
    with open(filename, 'w') as f:
        return f.write(contents)

def get_settings(filename: str) -> Settings:
    settings = Settings.from_json(read_whole_file(filename))
    settings.my_filename = filename
    return settings

@dataclass_json
@dataclass(frozen=False)
class Worker:
    name: str
    trainer: TrainingSettings
    lock: Optional[FileLock] = None
    log_file: str
    awake: bool = False
    woke_up: int = 0

    def log(self, thing):
        with self.lock.acquire():
            with open(self.log_file, 'a') as f:
                if isinstance(thing, str):
                    print(thing, flush=True)
                else:
                    pprint(thing)

    def epochs(self, settings: Settings) -> str:
        names = map('e_{}_{}'.format, count(), repeat(self.name))
        while time() < self.woke_up + settings.eon_lifetime:
            yield next(names)


@dataclass_json
@dataclass(frozen=True)
class State:
    workers: Dict[str, Worker]
    start_time: Optional[int] = None

@contextmanager
def initialize_and_recure(settings: Settings):
    lock = FileLock(settings.lock_file, settings.lock_timeout)
    with lock.acquire():
        state = State.from_json(read_whole_file(settings.state_file))
        if state.start_time is None:
            state.start_time = int(time())
        sleeping_workers = [w for w in state.workers.values() if not w.awake]
        next_profile = next(ts
                            for ts in settings.base_trainer_profiles
                            if ts not in {w.trainer for w in state.workers.values()},
                            random.choice(settings.base_trainer_profiles))
        worker = next(sleeping_workers,
                      Worker(name=f'w_{settings.lockfile.strip(".")}_{len(state.workers)}_{next_profile.name}',
                             trainer=next_profile,
                             log_file=settings.log_file))
        worker.lock = lock
        worker.awake = True
        worker.woke_up = int(time())
        state.workers[worker.name] = worker
        write_whole_file(settings.state_file, state.to_json())
    try:
        yield worker
    finally:
        with lock.acquire():
            state = State.from_json(read_whole_file(settings.state_file))
            worker.lock = None
            worker.awake = False
            state.workers[worker.name] = worker
            write_whole_file(settings.state_file, state.to_json())
        if time() < state.start_time + settings.total_lifetime:
            slurm_recurse()
            worker.log("recursing!")
        else:
            worker.log("dieing!")

def slurm_recurse():
    run(['sbatch', 'bates_cs354_a1_submit.sh'])



