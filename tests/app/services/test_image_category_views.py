"""Pins which image categories each gallery view lists.

`OTHER` is the private category for images a canvas layer owns — paint bitmaps,
composited layer pixels, adopted filter results. Those are layer pixels rather
than gallery content, so they must appear in neither the images view nor the
assets view, while still being non-intermediate so garbage collection cannot
strand the layer pointing at them.

Regression guarded: while `OTHER` was an assets category, every brush stroke and
every canvas operation added an image to the user's Assets tab.
"""

from invokeai.app.services.image_records.image_records_common import (
    ASSETS_CATEGORIES,
    IMAGE_CATEGORIES,
    ImageCategory,
    ResourceOrigin,
)
from invokeai.app.services.invocation_services import InvocationServices


def _save(services: InvocationServices, image_name: str, category: ImageCategory) -> None:
    services.image_records.save(
        image_name=image_name,
        image_origin=ResourceOrigin.INTERNAL,
        image_category=category,
        width=10,
        height=10,
        has_workflow=False,
    )


def test_other_belongs_to_neither_view() -> None:
    assert ImageCategory.OTHER not in IMAGE_CATEGORIES
    assert ImageCategory.OTHER not in ASSETS_CATEGORIES


def test_the_two_views_never_overlap() -> None:
    assert set(IMAGE_CATEGORIES).isdisjoint(ASSETS_CATEGORIES)


def test_assets_view_excludes_canvas_owned_images(mock_services: InvocationServices) -> None:
    _save(mock_services, "painted.png", ImageCategory.OTHER)
    _save(mock_services, "uploaded.png", ImageCategory.USER)

    result = mock_services.image_records.get_many(offset=0, limit=10, categories=ASSETS_CATEGORIES)

    assert [record.image_name for record in result.items] == ["uploaded.png"]


def test_images_view_excludes_canvas_owned_images(mock_services: InvocationServices) -> None:
    _save(mock_services, "painted.png", ImageCategory.OTHER)
    _save(mock_services, "generated.png", ImageCategory.GENERAL)

    result = mock_services.image_records.get_many(offset=0, limit=10, categories=IMAGE_CATEGORIES)

    assert [record.image_name for record in result.items] == ["generated.png"]


def test_date_board_counts_ignore_canvas_owned_images(mock_services: InvocationServices) -> None:
    """`get_image_dates` derives its counts from the category constants.

    It previously classified every non-`general` image as an asset, which would
    keep counting canvas-owned images after they left the assets view — leaving
    the date board's badge disagreeing with the images it actually lists.
    """
    _save(mock_services, "painted.png", ImageCategory.OTHER)
    _save(mock_services, "uploaded.png", ImageCategory.USER)
    _save(mock_services, "generated.png", ImageCategory.GENERAL)

    dates = mock_services.image_records.get_image_dates()

    assert len(dates) == 1
    assert dates[0].image_count == 1
    assert dates[0].asset_count == 1
