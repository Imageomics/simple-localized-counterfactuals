import os
import os.path as osp

import numpy as np
import pandas as pd
import torch
from torch.utils import data
from tqdm import tqdm


# Adapted from https://github.com/guillaumejs2403/ACE/blob/6684b93ef2d2d6a594fdbac87db25e3f640340e7/compute_CD.py
def get_correlations(path, query_label, HQ=False):
    if HQ:
        # read annotation files
        with open(osp.join(path, "list_attr_celeba.txt"), "r") as f:
            lines = f.readlines()[1:]
            cols = lines[0].split()
            data = [x.split() for x in lines[1:]]
            data_df = pd.DataFrame(data, columns=["image_id"] + cols)

        with open(osp.join(path, "image_list.txt"), "r") as f:
            lines = f.readlines()
            cols = lines[0].split()
            data = [x.split() for x in lines[1:]]
            mapping_df = pd.DataFrame(data, columns=cols)

        with open(osp.join(path, "list_eval_partition.txt"), "r") as f:
            lines = f.readlines()
            data = [x.split() for x in lines]
            partition_df = pd.DataFrame(data, columns=["image_id", "partition"])

        mapping_df.rename(columns={"orig_file": "image_id"}, inplace=True)
        partition_df = pd.merge(mapping_df, partition_df, on="image_id")
        partition_df.sort_values(by="image_id", inplace=True)
        data_df.sort_values(by="image_id", inplace=True)

        partition = "0"
        final_data_df = pd.merge(data_df, partition_df, on="image_id")
        df = final_data_df[final_data_df["partition"] == partition]
        df.reset_index(inplace=True)
        df.replace(-1, 0, inplace=True)
        labels = list(df.columns[1:])
        c = 2

    else:
        CELEBAPATH = os.path.join(path, "Anno", "list_attr_celeba.txt")
        CELEBAPATHP = os.path.join(path, "Eval", "list_eval_partition.txt")
        # extract the names of the labels

        with open(CELEBAPATH, "r") as f:
            lines = f.readlines()[1:]
            cols = lines[0].split()
            data = [x.split() for x in lines[1:]]
            df = pd.DataFrame(data, columns=["image_id"] + cols)

        with open(CELEBAPATHP, "r") as f:
            lines = f.readlines()
            data = [x.split() for x in lines]
            p = pd.DataFrame(data, columns=["image_id", "partition"])

        labels = list(df.columns[1:])
        df.sort_values(by="image_id", inplace=True)
        p.sort_values(by="image_id", inplace=True)
        df = df[p["partition"] == "0"]  # 1 is val, 0 train
        df.replace(-1, 0, inplace=True)
        c = 1

    corrs = np.zeros(40)

    for i in range(40):
        corrs[i] = np.corrcoef(
            df.iloc[:, query_label + c].to_numpy().astype(np.float32),
            df.iloc[:, i + c].to_numpy().astype(np.float32),
        )[0, 1]

    return corrs, labels


# Adapted from: https://github.com/guillaumejs2403/ACE/blob/6684b93ef2d2d6a594fdbac87db25e3f640340e7/compute_CD.py
@torch.no_grad()
def get_attrs_and_target_from_ds(paried_dataset, oracle, batch_size=32, num_workers=4):
    loader = data.DataLoader(
        paried_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    oracle_preds = {"cf": {"dist": [], "pred": []}, "cl": {"dist": [], "pred": []}}

    for cl, cf in tqdm(loader):
        cl = cl.to(dtype=torch.float).cuda()
        cf = cf.to(dtype=torch.float).cuda()

        cl_o_dist = oracle(cl)
        cf_o_dist = oracle(cf)

        oracle_preds["cl"]["dist"].append(cl_o_dist.cpu().numpy())
        oracle_preds["cl"]["pred"].append((cl_o_dist > 0.5).cpu().numpy())
        oracle_preds["cf"]["dist"].append(cf_o_dist.cpu().numpy())
        oracle_preds["cf"]["pred"].append((cf_o_dist > 0.5).cpu().numpy())

    oracle_preds["cl"]["dist"] = np.concatenate(oracle_preds["cl"]["dist"])
    oracle_preds["cf"]["dist"] = np.concatenate(oracle_preds["cf"]["dist"])
    oracle_preds["cl"]["pred"] = np.concatenate(oracle_preds["cl"]["pred"])
    oracle_preds["cf"]["pred"] = np.concatenate(oracle_preds["cf"]["pred"])

    return oracle_preds


# Adapted from: https://github.com/guillaumejs2403/ACE/blob/6684b93ef2d2d6a594fdbac87db25e3f640340e7/compute_CD.py
def compute_CorrMetric(
    oracle,
    paired_dataset,
    query_label,
    diff=True,
    remove_unchanged_oracle=False,
    batch_size=32,
    num_workers=4,
):
    oracle_preds = get_attrs_and_target_from_ds(
        paired_dataset, oracle, batch_size=batch_size, num_workers=num_workers
    )

    cf_pred = oracle_preds["cf"]["pred"].astype("float")
    cl_pred = oracle_preds["cl"]["pred"].astype("float")

    if diff:
        delta_query = cf_pred[:, query_label] - cl_pred[:, query_label]
        deltas = cf_pred - cl_pred
    else:
        delta_query = cf_pred[:, query_label]
        deltas = cf_pred

    if remove_unchanged_oracle:
        to_remove = cf_pred[:, query_label] != cl_pred[:, query_label]
        deltas = deltas[to_remove, :]
        delta_query = delta_query[to_remove]
        del to_remove

    print("Length:", len(deltas))

    our_corrs = np.zeros(40)

    for i in range(40):
        cc = np.corrcoef(deltas[:, i], delta_query)
        our_corrs[i] = 0 if np.any(np.isnan(cc)) else cc[0, 1]  # when a nan is found,

    return our_corrs
