import re
from contextlib import contextmanager

import pytest


@pytest.fixture(name="catch_parse_error")
def _catch_parse_error(capsys):
    @contextmanager
    def catch_parse_error(match: str = ""):
        try:
            yield
        except SystemExit:
            captured = capsys.readouterr()
            assert re.search(match, captured.err) is not None
        else:
            assert False, "Parse should fail"

    return catch_parse_error
