# RobLight
Artifact for the paper **Robustness Verification of Graph Neural Networks Via Lightweight Satisfiability Testing** by Chia-Hsuan Lu, Tony Tan, and Michael Benedikt, published at Tools and Algorithms for the Construction and Analysis of Systems (TACAS 2026), Turin, Italy.

```bibtex
@inproceedings{roblight2026,
    title  = {Robustness Verification of Graph Neural Networks Via Lightweight Satisfiability Testing},
    author = {Lu, Chia-Hsuan and 
              Tan, Tony and
              Benedikt, Michael},
    booktitle = {Tools and Algorithms for the Construction and Analysis of Systems},
    year = {2026},
    doi = {10.1007/978-3-032-22752-2_1}
}
```

## Build
To build our tool **RobLight**, run:
```bash
make
```
This will generate the executable file `roblight`.

## Run
### Run from command line
**RobLight** takes 12 command-line arguments:
```
./roblight <gnn_path> <graph_path> <feat_path> <output_path> <perturbation> <variant> <comp_radius> <gbudget> <lbudget> <ori> <shift> <timeout>
```

1. `<gnn_path>` -- Path to the Graph Neural Network (GNN) model file.
2. `<graph_path>` -- Path to the input graph structure file.
3. `<feat_path>` -- Path to the input graph feature file.
4. `<output_path>` -- File path where **RobLight** will save verification results. If set to empty (`""`), then **RobLight** will print the results to the terminal.
5. `<perturbation>` -- Type of perturbation to verify against:
    + `DEL_ONLY` -- deletion only.
    + `DEL_INS` -- deletion and insertion.
6. `<variant>` -- Variant of optimization strategies to use. This is a bitmask that combines the following options:
    + `1` -- disable incremental computation.
    + `2` -- disable reorder computation.
    + `4` -- disable bound tightening by budget.
    + `8` -- disable heuristic edge picking.
For example, if `<variant>` is 9, then **RobLight** will run without incremental computation and heuristic edge picking.
7. `<comp_radius>` -- Type of problem:
    + `0` -- robustness problem
    + `1` -- robustness radius problem.
8. `<gbudget>` -- Global budget.
9. `<lbudget>` -- Local budget.
10. `<ori>` -- The original class for the vertex. Use `-1` to specify the predicted class instead.
11. `<shift>` -- The shift for the weak robustness problem. Use `0` for the general robustness problem.
12. `<timeout>` — Maximum time (in seconds) allowed for verification. Use `0` for no timeout.

### Run from Python wrapper
We also provide a simple Python wrapper `run.py`.
```
usage: run.py [-h] --gnn-path GNN_PATH --graph-folder GRAPH_FOLDER
              [--mode {single,batch}] [--index INDEX] [--comp-radius]
              [--gbudget GBUDGET] [--lbudget LBUDGET] [--ori ORI]
              [--shift SHIFT] [--perturbation {DEL_ONLY,DEL_INS}]
              [--no-incremental] [--no-reorder] [--no-tight] [--no-heur-pick]
              [--timeout TIMEOUT] [--num-workers NUM_WORKERS]

RobLight Configuration

options:
  -h, --help            show this help message and exit
  --gnn-path GNN_PATH   Path to the GNN model file.
  --graph-folder GRAPH_FOLDER
                        Path to the graph folder.
  --mode {single,batch}
                        Solver mode: 'single' for one graph; 'batch' for
                        multiple graphs. Default: single.
  --index INDEX         For 'single' mode: index of the target graph. For
                        'batch' mode: process graphs with indices
                        0..(index-1). Default: 0.
  --comp-radius         Enable radius computation. If set, global/local
                        budgets are ignored.
  --gbudget GBUDGET     Global budget (ignored if --comp-radius is set).
                        Default: 1.
  --lbudget LBUDGET     Local budget (ignored if --comp-radius is set).
                        Default: same as global budget.
  --ori ORI             Original class. -1 indicates predicted class. Default:
                        predicted class.
  --shift SHIFT         Shift parameter for weak robustness. 0 indicates
                        general robustness. Default: 0.
  --perturbation {DEL_ONLY,DEL_INS}
                        Perturbation type: 'DEL_ONLY' (deletion only) or
                        'DEL_INS' (deletion and insertion). Default: DEL_ONLY.
  --no-incremental      Disable incremental computation when set.
  --no-reorder          Disable reordering computation when set.
  --no-tight            Disable tight bound computation when set.
  --no-heur-pick        Disable heuristic edge picking when set.
  --timeout TIMEOUT     Timeout per instance in seconds. Set 0 for no timeout.
                        Default: 300.
  --num-workers NUM_WORKERS
                        Number of graphs to solve in parallel. Default: 1 (no
                        parallelism).
```
In `single` mode, the result will be printed to the terminal. In `batch` mode, the results will be stored in `./data/log_verify/`.   Note that the following paths are relative to specific directories:
- `--gnn-path` is relative to `./data/models/`.
- `--graph-folder` is relative to `./data/graph/`.

For example, the following command will check the **general robustness** of the GNN model `Cora_max.gnnx` on the graph with the structure `Cora/2.graph` and features `Cora/2.feat` with global and local budgets `10`.
```
python3 run.py --gnn-path "Cora_max" --graph-folder "Cora" --index 2 --gbudget 10
```

## Input and Output Format
### GNN model file
- The **first line** indicates the type of GNN model -- either `NODE` or `GRAPH`.
- The **second line** specifies the aggregation function used in the GNN, which can be one of: `SUM`, `MAX`, or `MEAN`.
- The **third line** is an integer `l` representing the number of layers in the GNN (excluding the final pooling layer for graph classification).
- The **fourth line** contains `l + 1` integers for **node classification**, or `l + 2` integers for **graph classification**. These integers define the dimensionality of each layer. For graph classification, the last integer represents the dimension of the pooling layer.
- After these header lines, the file includes the coefficient matrices `C^(1)`, `A^(1)`, and bias vectors `b^(1)`, and so on for each layer. For graph classification, parameters for the pooling layer are also included.
    - Each matrix of size `r × m` is encoded using `r` lines, with each line containing `m` numbers separated by spaces.
    - Each vector of size `r` is encoded using `r` lines, with one number per line.

Here is an example file of a 4-layer **node classification** GNN with **sum** aggregation and dimensions of `1433`, `32`, `32`, `32`, and `7`:
```
NODE
SUM
4
1433 32 32 32 7
...
```

### Graph structure file
- The **first line** is an integer `n` representing the number of vertices in the graph.
- The **second line** indicates the **directionality** of the graph — either `DIRECTED` or `UNDIRECTED`.
- The following `n` lines represent the **incoming adjacency list** of each vertex. Each line starts with the number of incoming neighbors, followed by their vertex indices.

Here is an example file of a **directed** graph with 24 nodes, where node `0` has four incoming neighbors: `5`, `9`, `12`, and `20`:
```
24
DIRECTED
4 5 9 12 20
...
```

### Graph feature file
The feature file consists of `n` lines, where the `i`-th line represents the input feature vector of the `i`-th vertex.

### Output
The output consists of two parts. The first section is the configuration of **RobLight**, which includes:
- The paths to the GNN model file, graph structure file, and graph feature file.
- The type of problem: either the **radius problem**, or the **robustness problem** specified by the global and local budgets.
- The objective of the problem, defined by the original class and target class. If the target class is `other`, it indicates a general robustness problem.
- The type of perturbation — either `DELETION ONLY` or `DELETION INSERTION`.
- The variant of **RobLight** used. If this field is blank, all optimization strategies are enabled.

The second section is the status of **RobLight**, which includes:
- The result of the verification -- either `ROBUST`, `NONROBUST`, or `TIMEOUT`.
- The total runtime of **RobLight** in seconds.
- The structure of the recursive tree, including the number of nodes, cuts, and leaf nodes.
- Statistics showing the runtime ratio among **RobLight**’s three major components: non-robust tester, bound propagator, and graph manager. The value may be `nan` due to insufficient sampling.
- The computed radius of the problem (only applicable to radius problems).

Here is an example output:
```
========== configuration ==========
gnn_path:      ./data/models/Cora_max.gnnx
graph_path:    ./data/graph/Cora/2.graph
feature_path:  ./data/graph/Cora/2.feat
budgets (g/l): 10/10
objective:     1 > other
perturbation:  DELETION ONLY
variant:
============== status =============
result: ROBUST
time:   5.136445
#nodes: 102569
#cuts:  51285
#leafs: 5292
ratio:  0.00:0.75:0.25
```

## Test Cases and Additional Scripts

We provide several supporting resources and scripts:

+ The models used in the paper are stored in `./data/models`. The script `scripts/model.py` defines the architecture of the GNN models described in the paper.
+ The folder `./data/graph` contains only the first five graphs from each dataset. The complete set of graphs can be obtained by unzipping `./data/graph.zip` or by running the following commands (you may need to create a python virtual environment first):
```
pip install -r scripts/requirements.txt
python3 scripts/generate_graphs.py
```

## Instructions for Smoke Testing
For smoke testing, we provide a script that tests the robustness problem for both node and graph classification using the `sum`, `max`, and `mean` aggregation functions on the first five graphs of each dataset with budgets 5, as well as for radius computation.
To run the test, first build **RobLight** and unzip `./data/graph.zip`. Then, run:
```
python3 exp/smoke.py | tee smoke.log
```
A sample output is provided in `exp/smoke.out`. You can check the results by running:
```
diff smoke.log exp/smoke.out
```
Note that the `time` and `ratio` values may vary depending on the machine conditions.

## Instructions for Replication of Results
There are four experiments presented in the paper:
+ End-to-end performance on node classification (**exp1**).
+ Evaluation of optimization strategies (**exp2**).
+ Results on robustness radius (**exp3**).
+ End-to-end performance on graph classification (**exp4**).

To reproduce the results from the paper, follow the steps below.
1. Build **RobLight** and unzip `./data/graph.zip`.
2. Adjust the `num_workers` parameter in line 1 of `exp/exp.py`, which specifies the number of instances to be solved in parallel. For a fair comparison, we suggest setting it to no more than the number of available CPU cores.
3. Execute the following commands to reproduce the four main experiments:
    ```
    python3 exp/exp1.py
    python3 exp/exp2.py
    python3 exp/exp3.py
    python3 exp/exp4.py
    ```
    A smaller subset of experiments is also provided, covering only the Cornell, Texas, and MUTAG datasets. To run it:
    ```
    python3 exp/exp_subset.py
    ```
4. After all experiments have finished, run:
    ```
    python3 exp/summary.py
    ```
    This will print Tables 2–6 (as shown in the paper) to the terminal. Note that the number of solved instances may vary depending on machine conditions.

All experiments support resuming -- they can be safely stopped at any time and resumed later using the same command and configuration without changing any settings or arguments.

## Test Environment and Estimated Resource
The experiments reported in the paper were performed on a computing cluster equipped with Intel Xeon Platinum 8268 CPUs running CentOS 8. We also tested **RobLight** on the amd64 version of the TACAS virtual machine, hosted on a laptop with an Intel Core i5-1240P CPU running Arch Linux.

When utilizing 12 CPU cores, the estimated runtimes for the experiments are approximately 6 hours for **exp1**, 32 hours for **exp2**, 4 hours for **exp3**, and 7 hours for **exp4**. For the subset experiments, the estimated runtime is approximately 3 hours.

Note that **RobLight** relies on the x86 AVX2 instruction set extension. It requires that the host CPU, as well as any virtual CPUs in virtualized environments, support AVX2. As a result, **RobLight** is not compatible with arm64-based hosts or virtual machines.
