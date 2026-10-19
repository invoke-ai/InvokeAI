/**
 * Prefer the visible finished candidate, then the selected placeholder's live progress frame, otherwise no
 * preview.
 */

import type { CanvasPlacementContract, StagedPreviewInput } from '@workbench/canvas-engine/api';

/** Inputs to {@link selectStagedPreviewSource}. */
export interface StagedPreviewSelection {
  /** Whether the selected slot is a canvas placeholder that may have live progress. */
  isGenerationInFlight: boolean;
  /** The selected placeholder's denoise-progress frame, if any (b64 data URL + native dims). */
  progressImage: { dataUrl: string; width: number; height: number } | null;
  /** Whether finished candidate previews are visible. Placeholder progress is controlled separately. */
  isVisible: boolean;
  /** The currently selected staged candidate's image name, if any. */
  selectedImageName: string | null;
  /** The selected candidate's world-space placement, if it has one. */
  selectedPlacement?: CanvasPlacementContract | null;
  /** The current bbox size (document px) — progress frames scale to fill it. */
  bboxWidth: number;
  bboxHeight: number;
}

export const selectStagedPreviewSource = ({
  bboxHeight,
  bboxWidth,
  isGenerationInFlight,
  isVisible,
  progressImage,
  selectedImageName,
  selectedPlacement,
}: StagedPreviewSelection): StagedPreviewInput | null => {
  if (isVisible && selectedImageName) {
    return selectedPlacement
      ? { imageName: selectedImageName, placement: selectedPlacement }
      : { imageName: selectedImageName };
  }
  if (isGenerationInFlight && progressImage && bboxWidth > 0 && bboxHeight > 0) {
    // Progress frames are low-res latents of the bbox region; scale to fill it.
    return { dataUrl: progressImage.dataUrl, height: bboxHeight, width: bboxWidth };
  }
  return null;
};

/** Key staged preview inputs so decoding reruns for source/progress changes only. */
export const stagedPreviewKey = (source: StagedPreviewInput | null): string => {
  if (source === null) {
    return 'none';
  }
  if ('imageName' in source) {
    const { placement } = source;

    return placement
      ? `image:${source.imageName}:${placement.x}:${placement.y}:${placement.width}:${placement.height}:${placement.opacity}`
      : `image:${source.imageName}`;
  }
  return `data:${source.width}x${source.height}:${source.dataUrl}`;
};
