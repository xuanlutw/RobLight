from common import get_aggrs, get_node_datesets, get_graph_datesets, num_instance, get_node_ns, get_graph_ns, get_variants
from common import avg, wavg, sgm, er
from common import filter_TO, filter_NTO, filter_R, filter_RR, filter_NRNR, filter_RADIUS
from common import read_log, read_radius, read_graph

def comp_solved(ress, times):
    times = filter_NTO(times, ress)

    return len(times), avg(times), sgm(times)

def comp_timeout(ress, times):
    times = filter_TO(times, ress)

    return len(times), avg(times), sgm(times)

def comp_robust(ress, times):
    times = filter_R(times, ress)

    return len(times), avg(times), sgm(times)

def comp_robust2(ress1, ress2, times):
    times = filter_RR(times, ress1, ress2)

    return avg(times), sgm(times)

def comp_nonrobust2(ress1, ress2, times):
    times = filter_NRNR(times, ress1, ress2)

    return avg(times), sgm(times)

def comp_exp2(mask, times, nodes, es):
    times = [times[x] for x in range(len(times)) if mask[x]]
    nodes = [nodes[x] for x in range(len(times)) if mask[x]]
    es    = [es[x] for x in range(len(times)) if mask[x]]

    return avg(times), wavg(times, nodes) * 1000, er(nodes, es)

def comp_radius(ress, times, radiuss, r_max=10):
    nrs  = []
    tars = []
    tgrs = []
    for r in range(r_max+1):
        times_r = filter_RADIUS(times, ress, radiuss, r)
        nrs.append(len(times_r))
        tars.append(avg(times_r))
        tgrs.append(sgm(times_r))

    return nrs, tars, tgrs

def summary_exp1():
    print("[Table 2]")
    print(f"aggr      |   dataset |        |      all instances       |      robust instances")
    print(f"          |           |      # |      # |    t_a |    t_g |      # |    t_a |    t_g")
    print(f"------------------------------------------------------------------------------------")
    for dataset in get_node_datesets():
        ress = []
        times = []
        for n in get_node_ns():
            a, b, _ = read_log(dataset, "sum", n, n, shift=1)
            ress += a
            times += b

        na = len(ress)
        n,  ta,  tg  = comp_solved(ress, times)
        nr, tar, tgr = comp_robust(ress, times)

        print(f"sum(weak) | {dataset:>9} | {na:>6,} | {n:>6,} | {ta:6.2f} | {tg:6.2f} | {nr:>6,} | {tar:6.2f} | {tgr:6.2f}")
    print(f"------------------------------------------------------------------------------------")
    for aggr in get_aggrs():
        for dataset in get_node_datesets():
            ress = []
            times = []
            for n in get_node_ns():
                a, b, _ = read_log(dataset, aggr, n, n)
                ress += a
                times += b

            na = len(ress)
            n,  ta,  tg  = comp_solved(ress, times)
            nr, tar, tgr = comp_robust(ress, times)

            print(f"{aggr:<9} | {dataset:>9} | {na:>6,} | {n:>6,} | {ta:6.2f} | {tg:6.2f} | {nr:>6,} | {tar:6.2f} | {tgr:6.2f}")
        print(f"------------------------------------------------------------------------------------")

    print("[Table 3]")
    print(f"aggr      |   dataset | robust instances| nonrobust instances")
    print(f"          |           |    t_a |    t_g |    t_a |    t_g")
    print(f"---------------------------------------------------------")
    for dataset in get_node_datesets():
        ress_w, times_w, _ = read_log(dataset, "sum", 10, 10, shift=1)
        ress,   _,       _ = read_log(dataset, "sum", 10, 10)

        tar, tgr = comp_robust2(ress, ress_w, times_w)
        tan, tgn = comp_nonrobust2(ress, ress_w, times_w)

        print(f"sum(weak) | {dataset:>9} | {tar:6.2f} | {tgr:6.2f} | {tan:6.2f} | {tgn:6.2f}")
    print(f"---------------------------------------------------------")
    for dataset in get_node_datesets():
        ress_w, _,     _ = read_log(dataset, "sum", 10, 10, shift=1)
        ress,   times, _ = read_log(dataset, "sum", 10, 10)

        tar, tgr = comp_robust2(ress, ress_w, times)
        tan, tgn = comp_nonrobust2(ress, ress_w, times)

        print(f"sum(weak) | {dataset:>9} | {tar:6.2f} | {tgr:6.2f} | {tan:6.2f} | {tgn:6.2f}")
    print(f"---------------------------------------------------------")

def summary_exp2():
    print("[Table 4]")
    print(f"aggr |   dataset |         RobLight         |        w/o inc comp      |    w/o operator reorder  |   w/o bound tightening   |   w/o heuristic picking  |            w/o all")
    print(f"     |           |    t_a |    t_c |     ER |    t_a |    t_c |     ER |    t_a |    t_c |     ER |    t_a |    t_c |     ER |    t_a |    t_c |     ER |    t_a |    t_c |     ER")
    print(f"------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    for aggr in get_aggrs():
        for dataset in get_node_datesets():
            # prepare mask
            mask = [True for _ in range(num_instance(dataset))]
            for variant in get_variants(include_zero=True):
                ress, _, _ = read_log(dataset, aggr, 10, 10, variant=variant)
                if len(ress) > 0:
                    mask = [mask[x] and (ress[x] != "TO") for x in range(num_instance(dataset))]

            out = f"{aggr:<4} | {dataset:>9} "
            for variant in get_variants(include_zero=True):
                ress, times, nodes = read_log(dataset, aggr, 10, 10, variant=variant)
                _, es              = read_graph(dataset)

                ta, tac, er = comp_exp2(mask, times, nodes, es)
                out += f"| {ta:6.3f} | {tac:6.3f} | {er:6.2f} "
            print(out)
        print(f"------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")

def summary_exp3():
    print("[Table 5]")
    print(f"aggr |   dataset |    |robust instances |    r = 0    |    r = 1    |    r = 2    |    r = 3    |    r = 4    |    r = 5    |    r = 6    |    r = 7    |    r = 8    |    r = 9    |    r = 10   ")
    print(f"     |           |    |                 |   # |   t_a |   # |   t_a |   # |   t_a |   # |   t_a |   # |   t_a |   # |   t_a |   # |   t_a |   # |   t_a |   # |   t_a |   # |   t_a |   # |   t_a ")
    print(f"--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    for aggr in get_aggrs():
        for dataset in get_node_datesets():
            ress, times, radiuss, _ = read_radius(dataset, aggr)

            nt,  _,   _  = comp_timeout(ress, times)
            nr,  tar, _  = comp_robust(ress, times)
            nrs, tars, _ = comp_radius(ress, times, radiuss)

            out = f"{aggr:<4} | {dataset:>9} | {nt:>2,} | {nr:>6,} | {tar:6.2f} "
            for r in range(10 + 1):
                out += f"| {nrs[r]:>3,} | {tars[r]:6.2f}"
            print(out)
        print(f"--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")

def summary_exp4():
    print("[Table 6]")
    print(f"aggr      |   dataset |        |      all instances       |      robust instances")
    print(f"          |           |      # |      # |    t_a |    t_g |      # |    t_a |    t_g")
    print(f"------------------------------------------------------------------------------------")
    for dataset in get_graph_datesets():
        ress = []
        times = []
        for (ng, nl) in get_graph_ns():
            a, b, _ = read_log(dataset, "sum", ng, nl, shift=1, perturbation="DEL_INS")
            ress += a
            times += b

        na = len(ress)
        n,  ta,  tg  = comp_solved(ress, times)
        nr, tar, tgr = comp_robust(ress, times)

        print(f"sum(weak) | {dataset:>9} | {na:>6,} | {n:>6,} | {ta:6.2f} | {tg:6.2f} | {nr:>6,} | {tar:6.2f} | {tgr:6.2f}")
    print(f"------------------------------------------------------------------------------------")
    for aggr in get_aggrs():
        for dataset in get_graph_datesets():
            ress = []
            times = []
            for (ng, nl) in get_graph_ns():
                a, b, _ = read_log(dataset, aggr, ng, nl, perturbation="DEL_INS")
                ress += a
                times += b

            na = len(ress)
            n,  ta,  tg  = comp_solved(ress, times)
            nr, tar, tgr = comp_robust(ress, times)

            print(f"{aggr:<9} | {dataset:>9} | {na:>6,} | {n:>6,} | {ta:6.2f} | {tg:6.2f} | {nr:>6,} | {tar:6.2f} | {tgr:6.2f}")
        print(f"------------------------------------------------------------------------------------")

if __name__ == "__main__":
    summary_exp1()
    summary_exp2()
    summary_exp3()
    summary_exp4()
