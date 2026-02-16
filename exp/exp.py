num_workers = 12

import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common import get_aggrs, get_node_datesets, get_graph_datesets, num_instance
from common import get_node_ns, get_graph_ns, get_variants
from run import run_roblight

def run_exp1(do_subset=False):
    for dataset in get_node_datesets(do_subset):
        for n in get_node_ns():
            print(f"[weak robustness for sum GNN on {dataset} with budgets {n}]", flush=True)
            run_roblight("batch",
                         f"{dataset}_sum", dataset,
                         num_instance(dataset), "DEL_ONLY", 0,
                         False, n, n,
                         -1, 1,
                         300, num_workers)
    for aggr in get_aggrs():
        for dataset in get_node_datesets(do_subset):
            for n in get_node_ns():
                print(f"[robustness for {aggr} GNN on {dataset} with budgets {n}]", flush=True)
                run_roblight("batch",
                             f"{dataset}_{aggr}", dataset,
                             num_instance(dataset), "DEL_ONLY", 0,
                             False, n, n,
                             -1, 0,
                             300, num_workers)

def run_exp2(do_subset=False):
    for aggr in get_aggrs():
        for dataset in get_node_datesets(do_subset):
            for variant in get_variants():
                if (aggr == "max") and (variant == 2):
                    continue
                print(f"[robustness for {aggr} GNN on {dataset} with budgets 10 and variant {variant}]", flush=True)
                run_roblight("batch",
                             f"{dataset}_{aggr}", dataset,
                             num_instance(dataset), "DEL_ONLY", variant,
                             False, 10, 10,
                             -1, 0,
                             300, num_workers)

def run_exp3(do_subset=False):
    for aggr in get_aggrs():
        for dataset in get_node_datesets(do_subset):
            print(f"[radius computation for {aggr} GNN on {dataset}]", flush=True)
            run_roblight("batch",
                         f"{dataset}_{aggr}", dataset,
                         num_instance(dataset), "DEL_ONLY", 0,
                         True, 0, 0,
                         -1, 0,
                         300, num_workers)

def run_exp4(do_subset=False):
    for dataset in get_graph_datesets(do_subset):
        for (ng, nl) in get_graph_ns():
            
            print(f"[weak robustness for sum GNN on {dataset} with budgets {ng}/{nl}]", flush=True)
            run_roblight("batch",
                         f"{dataset}_sum", dataset,
                         num_instance(dataset), "DEL_INS", 0,
                         False, ng, nl,
                         -1, 1,
                         300, num_workers)
            print("", flush=True)
    for aggr in get_aggrs():
        for dataset in get_graph_datesets(do_subset):
            for (ng, nl) in get_graph_ns():
                print(f"[robustness for {aggr} GNN on {dataset} with budget {ng}/{nl}]", flush=True)
                run_roblight("batch",
                             f"{dataset}_{aggr}", dataset,
                             num_instance(dataset), "DEL_INS", 0,
                             False, ng, nl,
                             -1, 0,
                             300, num_workers)
                print("", flush=True)

if __name__ == "__main__":
    run_exp1()
    run_exp2()
    run_exp3()
    run_exp4()
