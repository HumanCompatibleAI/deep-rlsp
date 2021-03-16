import deep_rlsp
from imitation.scripts.train_adversarial import train_ex
from sacred.observers import FileStorageObserver
import os.path as osp


def main_console():
    observer = FileStorageObserver(osp.join("output", "sacred", "train"))
    train_ex.observers.append(observer)
    train_ex.run_commandline()


if __name__ == "__main__":  # pragma: no cover
    main_console()
