import argparse
import os

import networkx as nx
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--train-graph-path", default="",
                    type=str, required=True,
                    help="choose the prepared graph")
args = parser.parse_args()

G = nx.read_gpickle(args.train_graph_path)

ATOMIC_heads = list(
    set([(head, feat_dict["hid"]) for head, tail, feat_dict in G.edges.data() if feat_dict["relation"] == "ATOMIC"]))
ATOMIC_tails = list(set([tail for head, tail, feat_dict in G.edges.data() if feat_dict["relation"] == "ATOMIC"]))

print("num ATOMIC heads:", len(ATOMIC_heads))
print("num ATOMIC tails:", len(ATOMIC_tails))

ATOMIC_heads_dict = {}
for head, hid in ATOMIC_heads:
    if head in ATOMIC_heads_dict:
        ATOMIC_heads_dict[head].append(hid)
    else:
        ATOMIC_heads_dict[head] = [hid]
ATOMIC_tails_dict = dict([(node, True) for node in ATOMIC_tails])

inferred_npy = {"head": [], "tail": [], "new": []}
for head, tail, feat_dict in G.edges.data():
    if feat_dict["relation"] == "ASER":
        if head in ATOMIC_heads_dict:
            for hid in ATOMIC_heads_dict[head]:
                inferred_npy["head"].append((head, tail, hid))
        elif tail in ATOMIC_tails_dict:
            inferred_npy["tail"].append((head, tail))
        else:
            inferred_npy["new"].append((head, tail))
print(len(inferred_npy["head"]), len(inferred_npy["tail"]), len(inferred_npy["new"]))
np.save("infer-npy/" + os.path.basename(args.train_graph_path).split(".")[0], inferred_npy)
