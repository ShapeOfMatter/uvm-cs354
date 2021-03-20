from contextlib import contextmanager
from dataclasses import dataclass, asdict
from dataclasses_json import dataclass_json
from filelock import BaseFileLock, FileLock, SoftFileLock
from itertools import count, repeat
from pprint import pprint
import random
from subprocess import run
from time import time
import torch
from typing import Dict, Iterable, List, Optional, Tuple

@dataclass_json
@dataclass(frozen=True)
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
    trainable_features: Tuple[bool, ...]
    trainable_classifier: Tuple[bool, ...]
    training_profiles: Tuple[TrainingSettings, ...]
    max_batch_size: int
    max_seconds: int
    total_lifetime: int
    eon_lifetime: int

def read_whole_file(filename: str) -> str:
    with open(filename, 'r') as f:
        return f.read()

def write_whole_file(filename: str, contents: str) -> None:
    with open(filename, 'w') as f:
        f.write(contents)

def get_settings(filename: str) -> Settings:
    settings = Settings.from_json(read_whole_file(filename))
    settings.my_filename = filename
    return settings

@dataclass_json
@dataclass(frozen=True)
class _Worker:
    name: str
    trainer: TrainingSettings
    log_file: str
    awake: bool = False
    woke_up: int = 0

@dataclass(frozen=False)
class Worker:
    name: str
    trainer: TrainingSettings
    log_file: str
    lock: BaseFileLock = SoftFileLock("this_lock_should_not_exist", timeout=1)
    awake: bool = False
    woke_up: int = 0

    def as_dummy(self) -> _Worker:
        return _Worker(name=self.name, trainer=self.trainer, log_file=self.log_file, awake=self.awake, woke_up=self.woke_up)

    def log(self, thing):
        with self.lock.acquire():
            with open(self.log_file, 'a') as f:
                if isinstance(thing, str):
                    print(thing, flush=True)
                else:
                    pprint(thing)

    def epochs(self, settings: Settings) -> Iterable[str]:
        names = map('e_{}_{}'.format, count(), repeat(self.name))
        while time() < self.woke_up + settings.eon_lifetime:
            yield next(names)

    def upkeep_model(self, model, settings: Settings):
        with self.lock.acquire():
            state = State.from_json(read_whole_file(settings.state_file))
            if len(state.models) < settings.num_models:
                self.log('{self.name} is keeping the same model.')
            else:
                work_on = next(iter(sorted(state.models,
                                           key=lambda m: (len(m.worked_on),
                                                          len(m.failed_to_beat),
                                                          m.accuracy))))
                model.load_state_dict(torch.load(work_on.saved_in), strict=True)
                work_on.worked.append(self.name)
            write_whole_file(settings.state_file, state.to_json())

    def upkeep_state(self, model, accuracy, settings):
        with self.lock.acquire():
            state = State.from_json(read_whole_file(settings.state_file))
            if len(state.models) < settings.num_models:
                new_filename = settings.model_name_format.format(len(state.models))
                torch.save(model.state_dict(), new_filename)
                state.models.append(Model(accuracy=accuracy, worked_on=[], saved_in=new_filename))
                self.log('Saved a new model')
            else:
                models_by_ranking = sorted(state.models, key=lambda m: m.accuracy)
                current_worst = models_by_ranking[0]
                if accuracy < current_worst.accuracy:
                    self.log(f'Discarding model with accuracy {accuracy}.')
                else:
                    torch.save(model.state_dict(), current_worst.saved_in)
                    state.models = [*models_by_ranking[1:],
                                    Model(accuracy=accuracy, worked_on=[], saved_in=current_worst.saved_in)]
            if 1 < min(sum(name == w.name for name in m.worked_on)
                       for m in state.models
                       for w in state.workers.values()):
                self.trainer = use_next_profile(state, settings)
            write_whole_file(settings.state_file, state.to_json())

@dataclass_json
@dataclass(frozen=True)
class Model:
    accuracy: float
    worked_on: List[str]
    saved_in: str

@dataclass_json
@dataclass(frozen=True)
class State:
    workers: Dict[str, _Worker]
    models: List[Model]
    used_profiles: List[TrainingSettings]
    start_time: Optional[int] = None

def use_next_profile(state: State, settings: Settings) -> TrainingSettings:
    retval = next((ts for ts in settings.training_profiles if ts not in state.used_profiles),
                  settings.training_profiles[-1])
    state.used_profiles.append(retval)
    return retval

@contextmanager
def initialize_and_recure(settings: Settings):
    lock = FileLock(settings.lock_file, settings.lock_timeout)
    with lock.acquire():
        state = State.from_json(read_whole_file(settings.state_file))
        if state.start_time is None:
            state.start_time = int(time())
        sleeping_workers = [Worker(**asdict(w)) for w in state.workers.values() if not w.awake]
        worker = sleeping_workers[-1] if sleeping_workers else Worker(
            name=f'w_{settings.lock_file.strip(".")}_{len(state.workers)}',
            trainer=use_next_profile(state, settings),
            log_file=settings.log_file,
            lock=lock
        )
        worker.lock = lock
        worker.awake = True
        worker.woke_up = int(time())
        state.workers[worker.name] = worker.as_dummy()
        write_whole_file(settings.state_file, state.to_json())
    try:
        yield worker
    finally:
        with lock.acquire():
            state = State.from_json(read_whole_file(settings.state_file))
            worker.awake = False
            state.workers[worker.name] = worker.as_dummy()
            write_whole_file(settings.state_file, state.to_json())
        if time() < state.start_time + settings.total_lifetime:
            slurm_recurse()
            worker.log("recursing!")
        else:
            worker.log("dieing!")

def slurm_recurse():
    run(['sbatch', 'bates_cs354_a1_submit.sh'])



