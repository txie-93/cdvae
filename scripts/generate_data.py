import pandas as pd
from jarvis.db.figshare import data
import random
import numpy as np
from jarvis.core.atoms import Atoms


def get_id_train_val_test(
    total_size=1000,
    split_seed=123,
    train_ratio=None,
    val_ratio=0.1,
    test_ratio=0.1,
    n_train=None,
    n_test=None,
    n_val=None,
    keep_data_order=False,
):
    """Get train, val, test IDs."""
    if (
        train_ratio is None
        and val_ratio is not None
        and test_ratio is not None
    ):
        if train_ratio is None:
            assert val_ratio + test_ratio < 1
            train_ratio = 1 - val_ratio - test_ratio
            print("Using rest of the dataset except the test and val sets.")
        else:
            assert train_ratio + val_ratio + test_ratio <= 1
    # indices = list(range(total_size))
    if n_train is None:
        n_train = int(train_ratio * total_size)
    if n_test is None:
        n_test = int(test_ratio * total_size)
    if n_val is None:
        n_val = int(val_ratio * total_size)
    ids = list(np.arange(total_size))
    if not keep_data_order:
        random.seed(split_seed)
        random.shuffle(ids)
    # np.random.shuffle(ids)
    if n_train + n_val + n_test > total_size:
        raise ValueError(
            "Check total number of samples.",
            n_train + n_val + n_test,
            ">",
            total_size,
        )

    id_train = ids[:n_train]
    id_val = ids[-(n_val + n_test) : -n_test]  # noqa:E203
    id_test = ids[-n_test:]
    return id_train, id_val, id_test


tag = "jid"
dataset = "dft_2d"
prop = "exfoliation_energy"
dat = data(dataset)
mem = []
for i in dat:
    if i[prop] != "na":
        info = {}
        info["material_id"] = i[tag]
        info["cif"] = ((Atoms.from_dict(i["atoms"])).pymatgen_converter()).to(
            fmt="cif"
        )
        info["prop"] = i[prop]
        mem.append(info)


id_train, id_val, id_test = get_id_train_val_test(total_size=len(mem))
dataset_train = pd.DataFrame([mem[x] for x in id_train])
dataset_val = pd.DataFrame([mem[x] for x in id_val])
dataset_test = pd.DataFrame([mem[x] for x in id_test])

df_train = pd.DataFrame(dataset_train)
df_val = pd.DataFrame(dataset_val)
df_test = pd.DataFrame(dataset_test)

df_train.to_csv("train.csv")
df_val.to_csv("val.csv")
df_test.to_csv("test.csv")
