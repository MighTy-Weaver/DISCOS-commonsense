import argparse

import networkx as nx
import numpy as np
from tqdm import tqdm

from utils.atomic_utils import O_SUBJS
from utils.atomic_utils import get_ppn_substitue_dict

parser = argparse.ArgumentParser()
parser.add_argument("--relation", default='xWant', type=str, required=True,
                    choices=['oEffect', 'oReact', 'oWant', 'xAttr',
                             'xEffect', 'xIntent', 'xNeed', 'xReact', 'xWant'],
                    help="choose which relation to process")
parser.add_argument('--rm_want', action='store_true', default=True,
                    help="whether to remove want/intent/need")
args = parser.parse_args()

# 1. load ATOMIC attrs
ATOMIC_heads = np.load('../Matching-atomic/ASER-format-words/ATOMIC_head_words_withpersonz.npy', allow_pickle=True)
ATOMIC_xwant = np.load('../Matching-atomic/ASER-format-words-final/ATOMIC_tails_{}.npy'.format(args.relation),
                       allow_pickle=True)
clause_idx = np.load('../Matching-atomic/clause_idx.npy', allow_pickle=True)
wc_idx = np.load('../Matching-atomic/wildcard_idx.npy', allow_pickle=True)

G_atomic = nx.DiGraph()
# 2. Add ATOMIC edges
atomic_edges = []
for i, (head_list, tails_list) in tqdm(enumerate(zip(ATOMIC_heads, ATOMIC_xwant))):
    if i in clause_idx or i in wc_idx:
        continue
    for head in head_list:
        head_split = head.split()

        if len(head_split) == 0:
            continue
        # 1. find the position of all PPN, maximum 3
        head_subj = head_split[0]
        pronoun_dict = get_ppn_substitue_dict(head_split)
        head_c = " ".join([pronoun_dict.get(w, w) for w in head_split])

        # 2. Traverse tails_list
        for j, tail_list in enumerate(tails_list):
            tails_splits = [tail.split() for tail in tail_list if len(tail) > 0]
            tails_subjs = [tail_spl[0] for tail_spl in tails_splits]
            tails_c = [" ".join([pronoun_dict.get(w, w) for w in tail_spl]) \
                       for tail_spl in tails_splits]
            for tail_c, t_subj in zip(tails_c, tails_subjs):
                if args.relation in ["oWant", "oEffect", "oReact"]:
                    if t_subj != head_subj and t_subj in O_SUBJS:
                        if args.rm_want:
                            if args.relation == "oWant":
                                if "want to" in tail_c:
                                    continue
                        tail_c = " ".join(["PersonY"] + tail_c.split()[1:])
                        G_atomic.add_edge(head_c, tail_c,
                                          relation="ATOMIC", hid=i, tid=j)
                else:
                    if t_subj == head_subj:
                        if args.rm_want:
                            if args.relation == "xWant":
                                if "want to" in tail_c:
                                    continue
                            elif args.relation == "xIntent":
                                if "intent to" in tail_c:
                                    continue
                            elif args.relation == "xNeed":
                                if "need to" in tail_c:
                                    continue
                        G_atomic.add_edge(head_c, tail_c,
                                          relation="ATOMIC", hid=i, tid=j)
print("num ATOMIC edges:",
      sum([feat_dict["relation"] == "ATOMIC" for _, _, feat_dict in G_atomic.edges.data()]))

nx.write_gpickle(G_atomic, "graph-for-training/G_atomic_{}.pickle".format(args.relation))
