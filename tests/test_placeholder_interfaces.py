import pytest

from forecasting_system.agents.teacher import Teacher
from forecasting_system.exceptions import PlaceholderNotImplementedError
from forecasting_system.tools.evaluation import load_exports
from forecasting_system.tools.news import load_news


def test_placeholder_tools_raise_explicit_placeholder_error():
    with pytest.raises(PlaceholderNotImplementedError):
        load_news()
    with pytest.raises(PlaceholderNotImplementedError):
        load_exports()
