"""
    Input a relation, prepare
"""
import argparse
import os

import networkx as nx
import numpy as np
import pandas as pd

relations = ['oEffect', 'oReact', 'oWant', 'xAttr',
             'xEffect', 'xIntent', 'xNeed', 'xReact', 'xWant']

parser = argparse.ArgumentParser()
parser.add_argument("--relation", default='xWant', type=str, required=True,
                    choices=relations,
                    help="choose which relation to process")
parser.add_argument("--atomic_graph_path", default="graph-for-training/G_atomic_{}.pickle",
                    type=str, required=False,
                    help="choose the ATOMIC graph path")
parser.add_argument("--train_graph_path", default="graph-for-training/G.pickle",
                    type=str, required=True,
                    help="choose the ASER graph path")
parser.add_argument("--neg_prop", default=1.0,
                    type=float, required=False,
                    help="proportion of all neg samples")
parser.add_argument("--prop_other", default=20,
                    type=int, required=False,
                    help="proportion of negative examples from other relations")
parser.add_argument("--prop_inverse", default=10,
                    type=int, required=False,
                    help="proportion of negative examples from inversing head&tail")
parser.add_argument("--prop_atomic", default=10,
                    type=int, required=False,
                    help="proportion of negative examples from random (h, t) pairs,"
                         "where h \in H and t \in T")
args = parser.parse_args()

# load graph
atomic_paths = args.atomic_graph_path
atomic_graph_dict = dict([(r, nx.read_gpickle(atomic_paths.format(r))) \
                          for r in relations])
G_aser = nx.read_gpickle(args.train_graph_path)
G_aser = G_aser.copy()
neg_prop = args.neg_prop

if args.relation.startswith("x"):
    relations = ['xAttr', 'xEffect', 'xIntent', 'xNeed', 'xReact', 'xWant']
else:
    relations = ["oWant", "oReact", "oEffect"]
# split file
atomic_raw = pd.read_csv("/home/tfangaa/Downloads/ATOMIC/v4_atomic_all_agg.csv")
splits = dict((i, spl) for i, spl in enumerate(atomic_raw['split']))

num_dataset = {"trn": 0, "tst": 0, "dev": 0}
all_edges = {"trn": [], "tst": [], "dev": []}
for head, tail, feat_dict in G_aser.edges.data():
    if feat_dict["relation"] == "ATOMIC":
        spl = splits[feat_dict["hid"]]
        num_dataset[spl] += 1
        all_edges[spl].append((head, tail, feat_dict))
print('Number of positive training examples after trucating:{}, validating:{}, testing:{}'.format(num_dataset["trn"],
                                                                                                  num_dataset["dev"],
                                                                                                  num_dataset["tst"]))

# concate all the edges in other relations
alex_dict = {"PersonX": "alex", "PersonY": "bob", "PersonZ": "cindy"}
replace_alex = lambda strs: " ".join([alex_dict.get(tk, tk) for tk in strs.split()])
other_edges = {"trn": [], "tst": [], "dev": []}
for r in relations:
    if r != args.relation:
        for head, tail, feat_dict in atomic_graph_dict[r].edges.data():
            spl = splits[feat_dict["hid"]]
            other_edges[spl].append((replace_alex(head), replace_alex(tail), {"relation": "neg_" + spl}))
print("number of other edges. trn:{}, dev:{}, tst:{}".format(len(other_edges["trn"]), len(other_edges["dev"]),
                                                             len(other_edges["tst"])))

prop_other = args.prop_other / 100.0 * neg_prop
for spl in other_edges:
    idx = np.random.choice(list(range(len(other_edges[spl]))),
                           int(num_dataset[spl] * prop_other), replace=False)
    G_aser.add_edges_from([other_edges[spl][i] for i in idx])

# Add inverse neg samples
prop_inv = args.prop_inverse / 100.0 * neg_prop
for spl in all_edges:
    idx = np.random.choice(list(range(len(all_edges[spl]))),
                           int(num_dataset[spl] * prop_inv), replace=False)
    inv_edges = [all_edges[spl][i] for i in idx]
    inv_edges = [(tail, head, {"relation": "neg_" + spl}) for head, tail, _ in inv_edges]
    G_aser.add_edges_from(inv_edges)

prop_atomic = args.prop_atomic / 100.0 * neg_prop

for spl in all_edges:
    heads_set = list(set([head for head, tail, feat in all_edges[spl]]))
    tails_set = list(set([tail for head, tail, feat in all_edges[spl]]))
    edge_dict = dict([((head, tail), True) for head, tail, feat in all_edges[spl]])
    samp_num = int(num_dataset[spl] * prop_atomic)
    atomic_neg_edges = []
    for i in range(samp_num):
        head_id = np.random.randint(0, len(heads_set))
        tail_id = np.random.randint(0, len(tails_set))
        while (heads_set[head_id], tails_set[tail_id]) in edge_dict:
            head_id = np.random.randint(0, len(heads_set))
            tail_id = np.random.randint(0, len(tails_set))
        atomic_neg_edges.append((heads_set[head_id], tails_set[tail_id], {"relation": "neg_" + spl}))
    G_aser.add_edges_from(atomic_neg_edges)

print("num of prepared neg edges, train:{}, dev:{}, test:{}".format(
    len([1 for _, _, feat in G_aser.edges.data() if feat["relation"] == "neg_trn"]),
    len([1 for _, _, feat in G_aser.edges.data() if feat["relation"] == "neg_dev"]),
    len([1 for _, _, feat in G_aser.edges.data() if feat["relation"] == "neg_tst"]), )
)

save_name = os.path.join("graph-for-training",
                         os.path.basename(args.train_graph_path).split(".")[
                             0] + "_neg_{}_other_{}_inv_{}_atomic_{}.pickle".format(neg_prop, args.prop_other,
                                                                                    args.prop_inverse,
                                                                                    args.prop_atomic))
nx.write_gpickle(G_aser, save_name)
