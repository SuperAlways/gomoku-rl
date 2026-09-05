from gomoku.sgf import _coord, build_sgf


def test_coord_mapping():
    assert _coord(7, 7) == "hh"
    assert _coord(0, 0) == "aa"
    assert _coord(14, 14) == "oo"


def test_build_sgf_full():
    moves = [(1, 7, 7), (2, 7, 8), (1, 6, 6)]
    s = build_sgf("Alice", "Bob", "black_win", moves)
    assert s.startswith("(") and s.endswith(")")
    assert "FF[4]SZ[15]" in s
    assert "PB[Alice]" in s and "PW[Bob]" in s
    assert "RE[B+]" in s
    assert ";B[hh]" in s and ";W[hi]" in s and ";B[gg]" in s


def test_build_sgf_result_variants():
    assert "RE[W+]" in build_sgf("a", "b", "white_win", [])
    assert "RE[0]" in build_sgf("a", "b", "draw", [])
    assert "RE[" not in build_sgf("a", "b", "ongoing", [])