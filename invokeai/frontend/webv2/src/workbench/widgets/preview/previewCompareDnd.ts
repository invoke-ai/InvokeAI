import { isGalleryImageDragData } from '@features/gallery/utility';

/**
 * dnd-kit contract for the preview's drop-to-compare target: dropping any
 * gallery-image drag (gallery grid or preview filmstrip thumbs) onto the
 * preview frame arms it as the comparison image.
 */

export const PREVIEW_COMPARE_DROP_ID = 'preview-compare-target';

export interface PreviewCompareDropData {
  kind: 'preview-compare-target';
}

export const PREVIEW_COMPARE_DROP_DATA: PreviewCompareDropData = { kind: 'preview-compare-target' };

export const isPreviewCompareDropData = (value: unknown): value is PreviewCompareDropData =>
  typeof value === 'object' && value !== null && (value as PreviewCompareDropData).kind === 'preview-compare-target';

/** The image to arm for comparison, or null when the drag/drop pair is not ours. */
export const resolvePreviewCompareDrop = (activeData: unknown, overData: unknown): { imageName: string } | null => {
  if (!isGalleryImageDragData(activeData) || !isPreviewCompareDropData(overData)) {
    return null;
  }

  const imageName = activeData.images[0]?.imageName;

  return imageName ? { imageName } : null;
};
