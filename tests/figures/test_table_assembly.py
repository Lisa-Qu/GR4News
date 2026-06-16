from figures.assemble_main_tables import (
    loss_label_table_from_log,
    stars_for,
    fmt_cell,
)

# --- Task 3: loss×label parser ---
SAMPLE = '''
bce_binary       lambda=0.5  R@1=0.0361 R@5=0.1134 R@10=0.1720 R@50=0.3671
bce_soft         lambda=0.5  R@1=0.0352 R@5=0.1100 R@10=0.1700 R@50=0.3671
listmle_binary   lambda=0.5  R@1=0.0340 R@5=0.1050 R@10=0.1650 R@50=0.3671
'''


def test_loss_label_best_r1_per_config():
    t = loss_label_table_from_log(SAMPLE)
    assert t["bce"]["binary"] == 0.0361
    assert t["bce"]["soft"] == 0.0352
    assert t["listmle"]["binary"] == 0.0340


# Real run_listwise_scorer grid is space-separated columns: name λ R@1 R@5 R@10
GRID = '''
Method                       Lambda      R@1      R@5     R@10
--------------------------------------------------------------
vanilla                      —        0.0331   0.1112   0.1692
bce_binary                   0.02     0.0356   0.1120   0.1700
bce_binary                   0.10     0.0361   0.1134   0.1720
bce_soft                     0.10     0.0352   0.1100   0.1700
listmle_binary               0.10     0.0340   0.1050   0.1650
listmle_soft                 pure     0.0300   0.1000   0.1600
approx_ndcg_binary           0.05     0.0345   0.1090   0.1680
approx_ndcg_soft             0.05     0.0330   0.1070   0.1660
'''


def test_loss_label_from_grid_columns():
    t = loss_label_table_from_log(GRID)
    assert t["bce"]["binary"] == 0.0361      # max across λ sweep
    assert t["bce"]["soft"] == 0.0352
    assert t["listmle"]["binary"] == 0.0340
    assert t["listmle"]["soft"] == 0.0300    # "pure" λ row parsed
    assert t["approx_ndcg"]["binary"] == 0.0345
    assert "vanilla" not in t


# --- Task 5: significance-star + cell-format helpers ---
def test_stars_for():
    assert stars_for(0.004) == "**"     # <0.01
    assert stars_for(0.03) == "*"       # <0.05
    assert stars_for(0.2) == ""


def test_fmt_cell():
    assert fmt_cell(0.0331) == "0.0331"
    assert fmt_cell(0.0331, pct=8.9) == "0.0331 (+8.9%)"
