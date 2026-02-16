import os
import csv
import math

def get_aggrs():
    return ['sum', 'max', 'mean']

def get_node_datesets(do_subset=False):
    if do_subset:
        return ['Cornell', 'Texas']
    else:
        return ['Cora', 'CiteSeer', 'Cornell', 'Texas', 'Wisconsin']

def get_graph_datesets(do_subset=False):
    if do_subset:
        return ['MUTAG']
    else:
        return ['MUTAG', 'ENZYMES']

def num_instance(dataset):
    if dataset == "Cora":
        return 2708
    elif dataset == "CiteSeer":
        return 3312
    elif dataset == "Cornell":
        return 183
    elif dataset == "Texas":
        return 183
    elif dataset == "Wisconsin":
        return 251
    elif dataset == "MUTAG":
        return 188
    elif dataset == "ENZYMES":
        return 600
    else:
        raise ValueError("WTF")

def get_node_ns():
    return [1, 2, 5, 10]

def get_graph_ns():
    return [(1, 1), (2, 1), (2, 2), (5, 1), (5, 2), (5, 5)]

def get_variants(include_zero=False):
    if include_zero:
        return [0, 1, 2, 4, 8, 15]
    else:
        return [1, 2, 4, 8, 15]

def avg(times):
    if len(times) == 0:
        return math.nan
    return sum(times) / len(times)

def wavg(times, weights):
    if len(times) == 0:
        return math.nan
    return sum(times) / sum(weights)

def sgm(times, shift=10):
    if len(times) == 0:
        return math.nan
    log_sum = sum(math.log(t + shift) for t in times) / len(times)
    return math.exp(log_sum) - shift

def er(nodes, es):
    ers = []
    for i in range(len(nodes)):
        ers.append((math.log2(nodes[i]) / (es[i]+1)))
    return avg(ers)

def filter_TO(arr, ress):
    # assert(len(arr) == len(ress))
    return [arr[i] for i in range(len(ress)) if ress[i] == "TO"]

def filter_NTO(arr, ress):
    # assert(len(arr) == len(ress))
    return [arr[i] for i in range(len(ress)) if ress[i] != "TO"]

def filter_R(arr, ress):
    # assert(len(arr) == len(ress))
    return [arr[i] for i in range(len(ress)) if ress[i] == "R"]

def filter_RR(arr, ress1, ress2):
    # assert(len(arr) == len(ress1))
    # assert(len(arr) == len(ress2))
    return [arr[i] for i in range(len(ress1)) if (ress1[i] == "R") and (ress2[i] == "R")]

def filter_NRNR(arr, ress1, ress2):
    # assert(len(arr) == len(ress1))
    # assert(len(arr) == len(ress2))
    return [arr[i] for i in range(len(ress1)) if (ress1[i] == "NR") and (ress2[i] == "NR")]

def filter_RADIUS(arr, ress, radius, r):
    # assert(len(arr) == len(ress))
    # assert(len(arr) == len(radius))
    return [arr[i] for i in range(len(ress)) if (ress[i] == "NR") and (radius[i] == r)]

def read_log(dataset, aggr, gbudget, lbudget, shift=0, perturbation="DEL_ONLY", variant=0):
    uid = f"({dataset}_{aggr})_({dataset})_({gbudget}_{lbudget})_-1_{shift}_{perturbation}"
    if variant & 1:
        uid += "_NOINC"
    if variant & 2:
        uid += "_NOREORDER"
    if variant & 4:
        uid += "_NOTIGHT"
    if variant & 8:
        uid += "_NOHEURPICK"

    summary_path = os.path.join("./data/log_verify/", uid, "summary_all.csv")

    ress = []
    times = []
    nodes = []

    if os.path.exists(summary_path):
        with open(summary_path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                ress.append(row[0].strip())
                times.append(float(row[1]))
                nodes.append(int(row[2]))

    return ress, times, nodes

def read_radius(dataset, aggr, shift=0, perturbation="DEL_ONLY", variant=0):
    uid = f"({dataset}_{aggr})_({dataset})_RADIUS_-1_{shift}_{perturbation}"
    if variant & 1:
        uid += "_NOINC"
    if variant & 2:
        uid += "_NOREORDER"
    if variant & 4:
        uid += "_NOTIGHT"
    if variant & 8:
        uid += "_NOHEURPICK"

    summary_path = os.path.join("./data/log_verify/", uid, "summary_all.csv")

    ress = []
    times = []
    radiuss = []
    nodes = []

    if os.path.exists(summary_path):
        with open(summary_path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                ress.append(row[0].strip())
                times.append(float(row[1]))
                radiuss.append(float(row[2]))
                nodes.append(int(row[3]))

    return ress, times, radiuss, nodes

def read_graph(dataset):
    graph_path = f'./data/graph/{dataset}'
    idx = num_instance(dataset)

    nodes = []
    edges = []
    for i in range(idx):
        graph_file = os.path.join(graph_path, f'{i}.graph')

        with open(graph_file) as f:
            n = int(f.readline())
            f.readline() # DIRECTED

            e = 0
            for _ in range(n):
                e += int(f.readline().split(" ")[0])

        nodes.append(n)
        edges.append(e)

    return nodes, edges
