"""
    Input a relation, and parsed ATOMIC heads and tails,
    Return a ASER one-hop subgraph induced by the given heads and tails.
"""
import argparse
from itertools import chain

import networkx as nx
import numpy as np
from tqdm import tqdm

from utils.atomic_utils import SUBJS, ATOMIC_SUBJS, O_SUBJS
from utils.atomic_utils import get_ppn_substitue_dict

parser = argparse.ArgumentParser()
parser.add_argument("--relation", default='xWant', type=str, required=True,
                    choices=['oEffect', 'oReact', 'oWant', 'xAttr',
                             'xEffect', 'xIntent', 'xNeed', 'xReact', 'xWant'],
                    help="choose which relation to process")
parser.add_argument("--head-successor-thresh", default=-1, type=int, required=True,
                    help="the threshold of expand with two hop edges")
parser.add_argument("--retrieve_hop", default="one", type=str, required=True,
                    choices=["one", "all"],
                    help="whether retrieve one-hop or all of the aser-core")
parser.add_argument("--atomic-graph-path", default="graph-for-training/G_atomic_nowant.pickle",
                    type=str, required=True,
                    help="choose the ATOMIC graph path")

args = parser.parse_args()
assert args.relation in args.atomic_graph_path, "check consistency of relation"
if args.relation in ["xEffect", "xWant", "xReact"]:
    atomic_scenario = "effect_agent"
elif args.relation in ["xIntent", "xNeed"]:
    atomic_scenario = "cause_agent"
elif args.relation in ["oEffect", "oReact", "oWant"]:
    atomic_scenario = "effect_theme"
else:
    atomic_scenario = "stative"

# 1. load graph
node2id_dict = np.load("aser-graph-file/ASER_core_node2id.npy", allow_pickle=True)[()]
id2node_dict = dict([(node2id_dict[node], node) for node in node2id_dict])
G_aser = nx.read_gpickle("aser-graph-file/G_aser_{}.pickle".format(atomic_scenario))
min_succ_thres = args.head_successor_thresh
G_atomic = nx.read_gpickle(args.atomic_graph_path)

# 2. load ATOMIC attrs
ATOMIC_heads = np.load('../Matching-atomic/ASER-format-words/ATOMIC_head_words_withpersonz.npy', allow_pickle=True)
ATOMIC_xwant = np.load('../Matching-atomic/ASER-format-words-final/ATOMIC_tails_{}.npy'.format(args.relation),
                       allow_pickle=True)
clause_idx = np.load('../Matching-atomic/clause_idx.npy', allow_pickle=True)
wc_idx = np.load('../Matching-atomic/wildcard_idx.npy', allow_pickle=True)

# 3. prepare all ATOMIC heads/tails in the training
ATOMIC_heads_for_training = [heads for i, heads in enumerate(ATOMIC_heads) \
                             if (not i in clause_idx) and (not i in wc_idx)]
ATOMIC_xwant_for_training = [tails_list for i, tails_list in enumerate(ATOMIC_xwant) \
                             if (not i in clause_idx) and (not i in wc_idx)]

# 4. find the position of all the ATOMIC nodes in ASER
ATOMIC_head_vocab = [node for node in \
                     np.unique(list(chain(*ATOMIC_heads_for_training))) \
                     if len(node) > 0 and len(node.split()) > 2 and node.split()[0] in SUBJS]
ATOMIC_tail_vocab = np.unique(list(chain(
    *[list(chain(*tail_list)) for tail_list in ATOMIC_xwant_for_training])))
ATOMIC_tail_vocab = [node for node in ATOMIC_tail_vocab \
                     if len(node) > 0 and len(node.split()) > 2 and node.split()[0] in SUBJS]
print("num of unique ATOMIC heads:", len(ATOMIC_head_vocab))
print("num of unique ATOMIC tails:", len(ATOMIC_tail_vocab))
print("num of ATOMIC head&tail intersection:", len(set(ATOMIC_head_vocab).intersection(set(ATOMIC_tail_vocab))))

# 5. find the positions of ATOMIC heads and tails in ASER
ATOMIC_nodes = list(set(ATOMIC_head_vocab + ATOMIC_tail_vocab))
atomic_ids_in_aser = []
aser_node_dict = dict([(node, True) for node in G_aser.nodes()])
for atomic_node in ATOMIC_nodes:
    aser_id = node2id_dict.get(atomic_node, -1)
    if aser_id != -1 and aser_id in aser_node_dict:
        atomic_ids_in_aser.append(aser_id)
print("number of ATOMIC head&tails that have a matching in ASER:", len(atomic_ids_in_aser))
atomic_head_ids = []
for atomic_node in ATOMIC_head_vocab:
    aser_id = node2id_dict.get(atomic_node, -1)
    if aser_id != -1 and aser_id in aser_node_dict:
        atomic_head_ids.append(aser_id)
atomic_head_ids_dict = dict([(key, True) for key in atomic_head_ids])
print("number of ATOMIC heads that have a matching in ASER:{}, proportion:{}" \
      .format(len(atomic_head_ids_dict), len(atomic_head_ids_dict) / len(ATOMIC_head_vocab)))

# Check how many ATOMIC relationships can be grounded to ASER
total_atomic_edge = 0
matched_atomic_edge = 0
for i, (head_list, tails_list) in tqdm(enumerate(zip(ATOMIC_heads, ATOMIC_xwant))):
    if i in clause_idx or i in wc_idx:
        continue
    for j, tail_list in enumerate(tails_list):
        if len(tail_list) > 0:
            total_atomic_edge += 1
        for head in head_list:
            head_split = head.split()
            if len(head_split) == 0:
                continue
            if any(G_aser.has_edge(node2id_dict.get(head, -1), node2id_dict.get(tail, -1)) for tail in tail_list):
                matched_atomic_edge += 1
                break
print("proportion of ATOMIC edges grounded in ASER:", matched_atomic_edge / total_atomic_edge, total_atomic_edge,
      matched_atomic_edge)

# 6. find one hop neighbors
node_data = G_aser.nodes.data()

onehop_neighbors = []
expand_twohop_edges = []
expand_twohop_nodes = []
for node in atomic_ids_in_aser:
    node_subj = id2node_dict.get(node, "none").split()[0]
    succs = list(G_aser.successors(node))
    preds = list(G_aser.predecessors(node))
    onehops = succs + preds
    # If the relation is Effect_thems, then filter according to SUBJS here.
    if args.relation in ["oWant", "oEffect", "oReact"]:
        onehops = [n for n in onehops if id2node_dict.get(n, "none").split()[0] in O_SUBJS]
    onehop_neighbors.extend(onehops)
    # if the node is in the heads:
    if node in atomic_head_ids_dict:
        if len(succs) < min_succ_thres:
            # expand it with two hop edges
            twohop_succs = []
            for succ in succs:
                # find the two hop nodes
                twohop_succs.extend([(th_node,
                                      node_data[th_node]["freq"],
                                      {"cooccurance_time": min(G_aser[node][succ]["cooccurance_time"],
                                                               G_aser[succ][th_node]["cooccurance_time"])}) \
                                     for th_node in G_aser.successors(succ)])
            # if effect_agent, filter subj
            if args.relation in ["oWant", "oEffect", "oReact"]:
                twohop_succs = [(n, freq, feat_dict) \
                                for n, freq, feat_dict in twohop_succs \
                                if id2node_dict.get(n, "none").split()[0] in O_SUBJS \
                                and id2node_dict.get(n, "none").split()[0] != node_subj]
            for twohop_succ, _, feat_dict in sorted(twohop_succs, key=lambda x: x[1], reverse=True) \
                    [:min(min_succ_thres - len(succs), len(twohop_succs))]:
                expand_twohop_edges.append((node, twohop_succ, feat_dict))
                expand_twohop_nodes.append(twohop_succ)
# onehop_neighbors, expand_twohop_edges
print("number of onehop_neighbors:", len(set(onehop_neighbors)))
print("number of expanded two hop edges:", len(expand_twohop_edges))
print("number of expanded two hop nodes:", len(set(expand_twohop_nodes)))

# 7. expand subgraph
onehop_subgraph_nodes = list(set(onehop_neighbors + expand_twohop_nodes + atomic_ids_in_aser))
G_aser_onehop = G_aser.subgraph(onehop_subgraph_nodes)
if args.retrieve_hop == "one":
    G_aser_onehop_additive = G_aser_onehop.copy()
else:
    G_aser_onehop_additive = G_aser.copy()
G_aser_onehop_additive.add_edges_from(expand_twohop_edges)
print("number of nodes in G_aser_onehop:", len(G_aser_onehop_additive.nodes()),
      "number of edges:", len(G_aser_onehop_additive.edges()))


# 7.5 filter edges based on patterns
def check_pattern(pattern_list, selected_pattern_list):
    return any(pattern in selected_pattern_list for pattern in pattern_list)


all_nodes_feat = dict(G_aser_onehop_additive.nodes.data())
if args.relation in ["xReact", "xAttr", "oReact"]:
    selected_patterns = ['s-v-o-be-o', 's-be-o', 'spass-v', 's-v-o-be-a',
                         's-v-a', 's-v-be-a', 's-v-be-o', 's-be-a']
    edges2remove = []
    for head, tail, feat_dict in G_aser_onehop_additive.edges.data():
        if not check_pattern(all_nodes_feat[tail]["patterns"], selected_patterns):
            edges2remove.append((head, tail))
    print("number of edges to remove", len(edges2remove))
    G_aser_onehop_additive.remove_edges_from(edges2remove)

# 8. conceptualized


def get_conceptualized_graph(G):
    """
      c for conceptualized
    """
    G_c = nx.DiGraph()
    for head, tail, feat_dict in G.edges.data():
        head = id2node_dict[head]
        tail = id2node_dict[tail]
        head_split = head.split()
        tail_split = tail.split()
        head_subj = head_split[0]
        tail_subj = tail_split[0]
        # if effect_agent, filter subj
        if args.relation in ["oWant", "oEffect", "oReact"]:
            if head_subj == tail_subj:
                continue
            pronoun_dict = get_ppn_substitue_dict(head_split)
            head_c = " ".join([pronoun_dict.get(w, w) for w in head_split])
            tail_c = " ".join([pronoun_dict.get(w, w) for w in tail_split])
            tail_c = " ".join(["PersonY"] + tail_c.split()[1:])
        elif head_subj == tail_subj and head_subj in SUBJS:
            pronoun_dict = get_ppn_substitue_dict(head_split)
            head_c = " ".join([pronoun_dict.get(w, w) for w in head_split])
            tail_c = " ".join([pronoun_dict.get(w, w) for w in tail_split])
        else:
            continue
        if G_c.has_edge(head_c, tail_c):
            G_c.add_edge(head_c, tail_c,
                         relation="ASER",
                         cooccurance_time=G_c[head_c][tail_c]["cooccurance_time"] + feat_dict["cooccurance_time"])
        else:
            G_c.add_edge(head_c, tail_c, relation="ASER",
                         cooccurance_time=feat_dict["cooccurance_time"])
    return G_c


G_aser_onehop_conceptualized = get_conceptualized_graph(G_aser_onehop_additive)
print("num of nodes before conceptualization:", len(G_aser_onehop_additive.nodes()))
print("num of nodes after conceptualization:", len(G_aser_onehop_conceptualized.nodes()))
print("num of edges before conceptualization:", len(G_aser_onehop_additive.edges()))
print("num of edges after conceptualization:", len(G_aser_onehop_conceptualized.edges()))

# 9. adding ATOMIC edges
G_aser_onehop_conceptualized.add_edges_from(G_atomic.edges.data())
# print statistics
print("num ASER edges:",
      len([1 for _, _, feat in G_aser_onehop_conceptualized.edges.data() \
           if feat["relation"] == "ASER"]))
print("num ATOMIC edges:",
      len([1 for _, _, feat in G_aser_onehop_conceptualized.edges.data() \
           if feat["relation"] == "ATOMIC"]))

# 10. Alex Bob version
all_nodes = list(G_aser_onehop_conceptualized.nodes())
for node in all_nodes:
    if len(node.split()) <= 2:
        G_aser_onehop_conceptualized.remove_node(node)
alex_dict = {"PersonX": "alex", "PersonY": "bob", "PersonZ": "cindy"}
replace_alex = lambda strs: " ".join([alex_dict.get(tk, tk) for tk in strs.split()])
edge_data = G_aser_onehop_conceptualized.edges.data()
edge_data = [(replace_alex(item[0]), replace_alex(item[1]), item[2]) for item in tqdm(edge_data)]
G_atomic_aser_onehop_alex = nx.DiGraph()
G_atomic_aser_onehop_alex.add_edges_from(edge_data)
print("num of nodes:", len(G_atomic_aser_onehop_alex.nodes()),
      "num of edges:", len(G_atomic_aser_onehop_alex.edges()))
if args.retrieve_hop == "one":
    nx.write_gpickle(G_atomic_aser_onehop_alex,
                     "graph-for-training/G_aser_{}_1hop_thresh_{}.pickle".format(args.relation, min_succ_thres))
else:
    nx.write_gpickle(G_atomic_aser_onehop_alex,
                     "graph-for-training/G_aser_{}_all_twohop_thresh_{}.pickle".format(args.relation, min_succ_thres))
