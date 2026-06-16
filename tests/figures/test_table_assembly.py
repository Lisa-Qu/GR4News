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


# --- Task 5: significance-star + cell-format helpers ---
def test_stars_for():
    assert stars_for(0.004) == "**"     # <0.01
    assert stars_for(0.03) == "*"       # <0.05
    assert stars_for(0.2) == ""


def test_fmt_cell():
    assert fmt_cell(0.0331) == "0.0331"
    assert fmt_cell(0.0331, pct=8.9) == "0.0331 (+8.9%)"
