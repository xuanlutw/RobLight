import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from run import run_roblight

aggrs = ['sum', 'max', 'mean']
n = 5

def test_node():
    for aggr in aggrs:
        print(f"[general robustness for {aggr} GNN on Cora with budgets 5]", flush=True)
        for i in range(n):
            run_roblight("single",
                         f"Cora_{aggr}", "Cora",
                         i,
                         "DEL_ONLY", 0,
                         False, 5, 5,
                         -1, 0,
                         300, 1)
            print("", flush=True)

def test_graph():
    for aggr in aggrs:
        print(f"[general robustness for {aggr} GNN on MUTAG with budgets 1]", flush=True)
        for i in range(n):
            run_roblight("single",
                         f"MUTAG_{aggr}", "MUTAG",
                         i,
                         "DEL_INS", 0,
                         False, 1, 1,
                         -1, 0,
                         300, 1)
            print("", flush=True)

def test_radius():
    for aggr in aggrs:
        print(f"[radius computation for {aggr} GNN on Cora]", flush=True)
        for i in range(n):
            run_roblight("single",
                         f"Cora_{aggr}", "Cora",
                         i,
                         "DEL_ONLY", 0,
                         True, 0, 0,
                         -1, 0,
                         300, 1)
            print("", flush=True)

if __name__ == "__main__":
    test_node()
    test_graph()
    test_radius()
