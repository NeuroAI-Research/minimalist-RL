import numpy as np


class EpochLogger:
    def save_config(s, x):
        pass

    def log(s, x):
        pass

    def setup_pytorch_saver(s, x):
        pass

    def store(s, **kw):
        if "EpRet" in kw:
            print(kw)

    def save_state(s, *args):
        pass

    def log_tabular(s, *args, **kw):
        pass

    def dump_tabular(s):
        pass


def setup_pytorch_for_mpi():
    pass


def sync_params(x):
    pass


def mpi_avg_grads(x):
    pass


def mpi_fork(*args):
    pass


def mpi_avg(x):
    return x


def proc_id():
    return 1


def mpi_statistics_scalar(x, with_min_and_max=False):
    x = np.array(x, dtype=np.float32)
    if with_min_and_max:
        return x.mean(), x.std(), x.min(), x.max()
    return x.mean(), x.std()


def num_procs():
    return 1
