"""Marks for culling gallery media, alike for images and videos: a flag, a rating and a color label.

The flag is two columns. `starred` is the pick, older than the other marks; `rejected` is its opposite. A write that
sets either one sets the other false, and a CHECK keeps an item from being both.
"""

from collections.abc import Collection
from enum import Enum
from typing import Literal, NamedTuple, Optional, Union

from invokeai.app.util.metaenum import MetaEnum

MAX_RATING = 5
"""Ratings run from 0, unrated, to this."""


class ColorLabel(str, Enum, metaclass=MetaEnum):
    """A color label on a gallery item."""

    RED = "red"
    YELLOW = "yellow"
    GREEN = "green"
    BLUE = "blue"
    PURPLE = "purple"


class MarkFlag(str, Enum, metaclass=MetaEnum):
    """A gallery item's flag: picked (stored as `starred`), rejected, or neither."""

    PICK = "pick"
    NONE = "none"
    REJECT = "reject"


NO_COLOR_LABEL = "none"
# A color label, or "none" for an item without one: what a write sets and what a filter matches.
ColorLabelChoice = Union[ColorLabel, Literal["none"]]


class MarkChanges(NamedTuple):
    """The marks a write sets on every item it names. None leaves that mark as it is."""

    rating: Optional[int] = None
    flag: Optional[MarkFlag] = None
    color_label: Optional[ColorLabelChoice] = None


class MarkFilters(NamedTuple):
    """Which marks a listing keeps, combined with AND. An item matches a list when it has any of its values; None (or
    an empty list) does not filter."""

    flags: Optional[Collection[MarkFlag]] = None
    min_rating: Optional[int] = None
    max_rating: Optional[int] = None
    color_labels: Optional[Collection[ColorLabelChoice]] = None
