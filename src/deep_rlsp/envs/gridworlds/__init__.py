from .room import RoomEnv
from .apples import ApplesEnv
from .train import TrainEnv
from .batteries import BatteriesEnv

from .room_spec import ROOM_PROBLEMS
from .apples_spec import APPLES_PROBLEMS
from .train_spec import TRAIN_PROBLEMS
from .batteries_spec import BATTERIES_PROBLEMS

TOY_PROBLEMS = {
    "room": ROOM_PROBLEMS,
    "apples": APPLES_PROBLEMS,
    "train": TRAIN_PROBLEMS,
    "batteries": BATTERIES_PROBLEMS,
}

TOY_ENV_CLASSES = {
    "room": RoomEnv,
    "apples": ApplesEnv,
    "train": TrainEnv,
    "batteries": BatteriesEnv,
}

__all__ = [
    "RoomEnv",
    "ApplesEnv",
    "TrainEnv",
    "BatteriesEnv",
]
