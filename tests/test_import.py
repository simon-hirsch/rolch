# Author: Simon Hirsch
# A real basic test to check that the import does not fail.


def test_import():
    try:
        import ondil  # noqa

        failed = False
    except Exception:
        failed = True

    assert not failed, "Import failed with Exception."
