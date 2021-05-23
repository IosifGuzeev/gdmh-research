from utility.BruteForceOptimizer import get_bool_vectors


def test_bool_vectors_generation():
    vectors = get_bool_vectors(3)
    assert len(vectors) == 8
    expected_vectors = [
        (False, False, False),
        (False, False, True),
        (False, True, False),
        (False, True, True),
        (True, False, False),
        (True, False, True),
        (True, True, False),
        (True, True, True)
    ]
    for vector in expected_vectors:
        assert vector in vectors
