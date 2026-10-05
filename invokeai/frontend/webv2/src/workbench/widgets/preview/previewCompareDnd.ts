import { isGalleryImageDragData } from '@features/gallery/utility';

export const PREVIEW_COMPARE_DROP_ID = 'preview-compare-target';

export interface PreviewCompareDropData {
  kind: 'preview-compare-target';
}

export const PREVIEW_COMPARE_DROP_DATA: PreviewCompareDropData = { kind: 'preview-compare-target' };

export const isPreviewCompareDropData = (value: unknown): value is PreviewCompareDropData =>
  typeof value === 'object' && value !== null && (value as PreviewCompareDropData).kind === 'preview-compare-target';

/**
 * Reject foreign drops and self-comparison, which would pause follow despite displaying nothing. Require callers
 * to provide currentImageName or null.
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
