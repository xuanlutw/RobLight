import argparse
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

bin_release_path = "./roblight"

model_path  = "./data/models"
graph_path  = "./data/graph"

log_verify_path = "./data/log_verify"

def prepare_cmd(binary,
                gnn_path, graph_folder, output_path,
                index,
                perturbation, variant,
                comp_radius, gbudget, lbudget,
                ori, shift,
                timeout, no_timeout=False):
    return [
        binary,
        os.path.join(model_path, f"{gnn_path}.gnnx"),
        os.path.join(graph_path, graph_folder, f"{index}.graph"),
        os.path.join(graph_path, graph_folder, f"{index}.feat"),
        output_path,
        perturbation,
        str(variant),
        str(1 if comp_radius else 0),
        str(gbudget), str(lbudget),
        str(ori), str(shift),
        str(0 if no_timeout else timeout)
    ]

def run_roblight(mode,
                 gnn_path, graph_folder,
                 index,
                 perturbation, variant,
                 comp_radius, gbudget, lbudget,
                 ori, shift,
                 timeout, num_workers):
    # prepare uid
    if comp_radius:
        uid = f"({gnn_path})_({graph_folder})_RADIUS_{ori}_{shift}_{perturbation}"
    else:
        uid = f"({gnn_path})_({graph_folder})_({gbudget}_{lbudget})_{ori}_{shift}_{perturbation}"
    if variant & 1:
        uid += "_NOINC"
    if variant & 2:
        uid += "_NOREORDER"
    if variant & 4:
        uid += "_NOTIGHT"
    if variant & 8:
        uid += "_NOHEURPICK"

    # run roblight
    if mode == "single":
        cmd = prepare_cmd(bin_release_path,
                          gnn_path, graph_folder, "",
                          index,
                          perturbation, variant,
                          comp_radius, gbudget, lbudget,
                          ori, shift,
                          timeout)
        subprocess.check_call(cmd)

    elif mode == "batch":
        output_path_base = os.path.join(log_verify_path, uid)
        os.makedirs(output_path_base, exist_ok=True)

        def run_cmd(i, cmd):
            subprocess.run(cmd)
            return i

        # task apply
        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            futures = []
            for i in range(index):
                output_path = os.path.join(output_path_base, f"{i}.log")
                if os.path.exists(output_path):
                    continue

                cmd = prepare_cmd(bin_release_path,
                                  gnn_path, graph_folder, output_path,
                                  i,
                                  perturbation, variant,
                                  comp_radius, gbudget, lbudget,
                                  ori, shift,
                                  timeout)
                futures.append(
                    ex.submit(run_cmd, i, cmd)
                )

            for f in as_completed(futures):
                print(f"{f.result()} done")

        # generate summary
        with (
            open(os.path.join(output_path_base, "summary.csv"), "w") as fp,
            open(os.path.join(output_path_base, "summary_all.csv"), "w") as fp_all
        ):
            for i in range(index):
                with open(os.path.join(output_path_base, f"{i}.log"), "r") as fp_in:
                    lines   = fp_in.readlines()
                    result  = " R" if lines[9][8] == "R" else ("NR" if lines[9][8] == "N" else "TO")
                    result += f", {lines[10][8:].strip()}"
                    if comp_radius:
                        result += f", {lines[15][8:].strip()}"

                    fp.write(result + "\n")

                    result += f", {lines[11][8:].strip()}"
                    result += f", {lines[12][8:].strip()}"
                    result += f", {lines[13][8:].strip()}"
                    result += f", {lines[14][8:].strip()}"

                    fp_all.write(result + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RobLight Configuration")

    parser.add_argument(
        "--gnn-path", type=str, required=True,
        help="Path to the GNN model file."
    )
    parser.add_argument(
        "--graph-folder", type=str, required=True,
        help="Path to the graph folder."
    )
    parser.add_argument(
        "--mode",
        choices=['single', 'batch'],
        default='single',
        help=("Solver mode: 'single' for one graph; 'batch' for multiple graphs. "
              "Default: single.")
    )
    parser.add_argument(
        "--index", type=int, default=0,
        help=("For 'single' mode: index of the target graph. "
              "For 'batch' mode: process graphs with indices 0..(index-1). "
              "Default: 0.")
    )
    parser.add_argument(
        "--comp-radius", action="store_true",
        help="Enable radius computation. If set, global/local budgets are ignored."
    )
    parser.add_argument(
        "--gbudget", type=int, default=1,
        help=("Global budget (ignored if --compute-radius is set). "
              "Default: 1.")
    )
    parser.add_argument(
        "--lbudget", type=int,
        help=("Local budget (ignored if --compute-radius is set). "
              "Default: same as global budget.")
    )
    parser.add_argument(
        "--ori", type=int, default=-1,
        help=("Original class. -1 indicates predicted class. "
              "Default: predicted class.")
    )
    parser.add_argument(
        "--shift", type=int, default=0,
        help=("Shift parameter for weak robustness. 0 indicates general robustness. "
              "Default: 0.")
    )
    parser.add_argument(
        "--perturbation", type=str, choices=['DEL_ONLY', 'DEL_INS'], default='DEL_ONLY',
        help=("Perturbation type: 'DEL_ONLY' (deletion only) or 'DEL_INS' (deletion and insertion). "
              "Default: DEL_ONLY.")
    )
    parser.add_argument(
        "--no-incremental", action='store_true',
        help="Disable incremental computation when set."
    )
    parser.add_argument(
        "--no-reorder", action='store_true',
        help="Disable reordering computation when set."
    )
    parser.add_argument(
        "--no-tight", action='store_true',
        help="Disable tight bound computation when set."
    )
    parser.add_argument(
        "--no-heur-pick", action='store_true',
        help="Disable heuristic edge picking when set."
    )
    parser.add_argument(
        "--timeout", type=int, default=300,
        help=("Timeout per instance in seconds. Set 0 for no timeout. "
              "Default: 300.")
    )
    parser.add_argument(
        "--num-workers", type=int, default=1,
        help=("Number of graphs to solve in parallel. "
              "Default: 1 (no parallelism).")
    )

    args = parser.parse_args()

    if not args.lbudget:
        args.lbudget = args.gbudget

    variant = 0
    if args.no_incremental:
        variant += 1
    if args.no_reorder:
        variant += 2
    if args.no_tight:
        variant += 4
    if args.no_heur_pick:
        variant += 8

    run_roblight(args.mode,
                 args.gnn_path, args.graph_folder,
                 args.index,
                 args.perturbation, variant,
                 args.comp_radius, args.gbudget, args.lbudget,
                 args.ori, args.shift,
                 args.timeout, args.num_workers)
