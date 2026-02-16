#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "Solver.h"
#include "utils.h"

int main(int argc, char *argv[]) {
    CHECK(argc != 13, "wrong number of arguments!");

    // read command line arguments
    const char *gnn_path    = argv[1];
    const char *graph_path  = argv[2];
    const char *feat_path   = argv[3];
    const char *output_path = argv[4];
    char       *pert        = argv[5];
    size_t      variant     = atoi(argv[6]);
    bool        comp_radius = atoi(argv[7]);
    size_t      gbudget     = atoi(argv[8]);
    size_t      lbudget     = atoi(argv[9]);
    size_t      ori         = atoi(argv[10]);
    size_t      shift       = atoi(argv[11]);
    double      timeout     = atof(argv[12]);

    // check robustness
    Solver *solver =
        Solver_alloc(gnn_path, graph_path, feat_path, pert, variant);
    Solver_check_robust(solver, comp_radius, gbudget, lbudget, ori, shift,
                        timeout);

    // write results
    FILE *fp;
    if (strcmp(output_path, "") == 0)
        fp = stdout;
    else
        fp = XFOPEN(output_path, "w");
    fprintf(fp, "========== configuration ==========\n");
    fprintf(fp, "gnn_path:      %s\n", gnn_path);
    fprintf(fp, "graph_path:    %s\n", graph_path);
    fprintf(fp, "feature_path:  %s\n", feat_path);
    if (comp_radius)
        fprintf(fp, "radius computation\n");
    else
        fprintf(fp, "budgets (g/l): %ld/%ld\n", gbudget, lbudget);
    fprintf(fp, "objective:     %ld > ", Solver_ori(solver));
    if (shift == 0)
        fprintf(fp, "other\n");
    else
        fprintf(fp, "%ld\n", Solver_tgt(solver));
    fprintf(fp, "perturbation:  ");
    Solver_dump_perturbation(solver, fp);
    fprintf(fp, "variant:       ");
    Solver_dump_variant(solver, fp);

    fprintf(fp, "============== status =============\n");
    fprintf(fp, "result: %s\n", Solver_result(solver));
    fprintf(fp, "time:   %lf\n", Solver_time(solver));
    fprintf(fp, "#nodes: %ld\n", Solver_num_nodes(solver));
    fprintf(fp, "#cuts:  %ld\n", Solver_num_cuts(solver));
    fprintf(fp, "#leafs: %ld\n", Solver_num_leafs(solver));
    fprintf(fp, "ratio:  %.2lf:%.2lf:%.2lf\n", Solver_ratio_feat(solver),
            Solver_ratio_bound(solver), Solver_ratio_pgraph(solver));
    if (comp_radius)
        fprintf(fp, "radius: %ld\n", Solver_radius(solver));

    // clean up
    if (fp != stdout)
        fclose(fp);
    Solver_free(solver);
    return 0;
}
