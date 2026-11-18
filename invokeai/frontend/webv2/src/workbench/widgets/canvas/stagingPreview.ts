/**
 * Prefer the selected finished candidate, then the selected placeholder's live progress frame, otherwise no
 * preview. Both land at the rect pinned at submission, so moving the bbox mid-run never moves them. A candidate the
 * eye toggled off stays selected as a hidden preview: the canvas under it compares without the masks the shown
 * candidate hides.
 */

import type { CanvasPlacementContract, Rect, StagedPreviewInput } from '@workbench/canvas-engine/api';

/** Inputs to {@link selectStagedPreviewSource}. */
export interface StagedPreviewSelection {
  /** Where the selected placeholder's result lands (its submitted rect); null when no placeholder is selected. */
  progressRect: Rect | null;
  /** The selected placeholder's denoise-progress frame, if any (b64 data URL + native dims). */
  progressImage: { dataUrl: string; width: number; height: number } | null;
  /** Whether finished candidate previews are visible. Placeholder progress is controlled separately. */
  isVisible: boolean;
  /** The selected staged candidate, if any, with its document-space placement. */
  selectedCandidate: { imageName: string; placement: CanvasPlacementContract } | null;
}

export const selectStagedPreviewSource = ({
  isVisible,
  progressImage,
  progressRect,
  selectedCandidate,
}: StagedPreviewSelection): StagedPreviewInput | null => {
  if (selectedCandidate) {
    const { imageName, placement } = selectedCandidate;
    return isVisible ? { imageName, placement } : { hidden: true, imageName, placement };
  }
  if (progressRect && progressImage && progressRect.width > 0 && progressRect.height > 0) {
    // Progress frames are low-res latents of the submitted region; scale to fill it.
    return { dataUrl: progressImage.dataUrl, placement: { ...progressRect, opacity: 1 } };
  }
  return null;
};

/** Key staged preview inputs so decoding reruns for source/progress changes only. */
export const stagedPreviewKey = (source: StagedPreviewInput | null): string => {
  if (source === null) {
    return 'none';
  }
  const { placement } = source;
  const rect = `${placement.x}:${placement.y}:${placement.width}:${placement.height}:${placement.opacity}`;
  if ('imageName' in source) {
    return `${source.hidden ? 'hidden' : 'image'}:${source.imageName}:${rect}`;
  }
  return `data:${rect}:${source.dataUrl}`;
};
