# Author: Simon Hirsch
# A real basic test to check that the import does not fail.


def test_import():
    try:
        import rolch

        failed = False
    except Exception as e:
        failed = True

    assert ~failed, f"Import failed with Exception {e}."
