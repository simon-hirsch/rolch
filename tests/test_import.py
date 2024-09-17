# Author: Simon Hirsch
# A real basic test to check that the import does not fail.


def test_import():
    try:
        import rolch  # noqa

        failed = False
    except Exception:
        failed = True

    assert not failed, f"Import failed with Exception."
