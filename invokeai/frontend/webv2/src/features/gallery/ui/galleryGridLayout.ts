export const GALLERY_GRID_GAP_PX = 4;

const GALLERY_MIN_COLUMN_COUNT = 2;
const GALLERY_MAX_COLUMN_COUNT = 12;

/** Cell size the density slider interpolates between: 0% is largest, 100% smallest. */
const GALLERY_MAX_CELL_PX = 192;
const GALLERY_MIN_CELL_PX = 48;

/**
 * Density picks a target thumbnail size, and the available width decides how
 * many of those fit.
 *
 * Keying off width rather than placement is what lets the two layouts share
 * one grid: at equal pixel width the same density produces the same columns
 * whether the gallery sits in a side panel or the center. The old
 * layout-keyed maximum meant 50% density meant ~90px cells stacked and ~200px
 * cells wide.
 */
export const getGalleryTargetCellPx = (imageDensityPercent: number): number => {
  const percent = Math.min(100, Math.max(0, imageDensityPercent));

  return GALLERY_MAX_CELL_PX - ((GALLERY_MAX_CELL_PX - GALLERY_MIN_CELL_PX) * percent) / 100;
};

export const getGalleryColumnCount = ({
  imageDensityPercent,
  widthPx,
}: {
  imageDensityPercent: number;
  widthPx: number;
}): number => {
  if (widthPx <= 0) {
    return GALLERY_MIN_COLUMN_COUNT;
  }

  const columnCount = Math.round(widthPx / getGalleryTargetCellPx(imageDensityPercent));

  return Math.min(GALLERY_MAX_COLUMN_COUNT, Math.max(GALLERY_MIN_COLUMN_COUNT, columnCount));
};

/** Falls back to a plausible square before the viewport has been measured. */
export const getGalleryCellSizePx = ({ columnCount, widthPx }: { columnCount: number; widthPx: number }): number =>
  widthPx > 0 ? Math.max(1, (widthPx - GALLERY_GRID_GAP_PX * (columnCount - 1)) / columnCount) : 96;
