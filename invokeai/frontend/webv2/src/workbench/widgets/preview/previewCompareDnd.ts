import { isGalleryImageDragData } from '@features/gallery/utility';

/**
 * dnd-kit contract for the preview's drop-to-compare target: dropping any
 * all-image gallery-item drag (gallery grid or preview filmstrip thumbs) onto the
 * preview frame arms it as the comparison image.
 */

export const PREVIEW_COMPARE_DROP_ID = 'preview-compare-target';

export interface PreviewCompareDropData {
  kind: 'preview-compare-target';
}

export const PREVIEW_COMPARE_DROP_DATA: PreviewCompareDropData = { kind: 'preview-compare-target' };

export const isPreviewCompareDropData = (value: unknown): value is PreviewCompareDropData =>
  typeof value === 'object' && value !== null && (value as PreviewCompareDropData).kind === 'preview-compare-target';

/**
 * The image to arm for comparison, or null when the drag/drop pair is not ours
 * — or when the drop would compare the previewed image with itself. That is not
 * merely a no-op: arming a comparison pauses live-follow, so a self-drop would
 * quietly switch off in-progress images and leave a comparison that springs
 * open on the next selection.
 *
 * `currentImageName` is required rather than optional so every caller has to
 * say what is on screen; a caller with nothing selected passes null.
 */
export const resolvePreviewCompareDrop = (
  activeData: unknown,
  overData: unknown,
  currentImageName: string | null
): { imageName: string } | null => {
  if (!isGalleryImageDragData(activeData) || !isPreviewCompareDropData(overData)) {
    return null;
  }

  const imageName = activeData.items[0]?.name;

  return imageName && imageName !== currentImageName ? { imageName } : null;
};
