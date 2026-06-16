from figures.lambda_log_parser import parse_heldout_log

SAMPLE = '''
Selecting Focal lambda on held-out (R@1 argmax over grid)...
  lambda=0.0   held-out R@1=0.0352
  lambda=0.1   held-out R@1=0.0375
  -> best Focal lambda=0.1 (held-out R@1=0.0375)
Selecting Listwise lambda on held-out (mean R@1 across seeds, same grid)...
  lambda=0.0   held-out mean R@1=0.0352
  lambda=0.1   held-out mean R@1=0.0378
  -> best Listwise lambda=0.1 (held-out mean R@1=0.0378)
Scoring Efficiency: vanilla=9.0%  best_scorer=9.6%
'''


def test_parse():
    r = parse_heldout_log(SAMPLE)
    assert r["focal"][0.0] == 0.0352 and r["focal"][0.1] == 0.0375
    assert r["listwise"][0.1] == 0.0378
    assert r["best_focal_lambda"] == 0.1 and r["best_listwise_lambda"] == 0.1
    assert abs(r["se_vanilla"] - 0.09) < 1e-9


def test_parse_amazon_se_format():
    """Amazon settings print a different SE line format (capital, comma sep)."""
    txt = '''
Selecting Listwise lambda on held-out (mean R@1 across seeds, same grid)...
  lambda=0.0   held-out mean R@1=0.0277
  lambda=0.02  held-out mean R@1=0.0309
  -> best Listwise lambda=0.02 (val mean R@1=0.0309)
Scoring Efficiency: Vanilla=15.2%, Best Scorer=15.9%
'''
    r = parse_heldout_log(txt)
    assert r["best_listwise_lambda"] == 0.02
    assert abs(r["se_vanilla"] - 0.152) < 1e-9
    assert abs(r["se_scorer"] - 0.159) < 1e-9
