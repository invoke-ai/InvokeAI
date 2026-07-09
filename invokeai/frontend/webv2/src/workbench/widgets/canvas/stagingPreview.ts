/**
 * Pure selection of what the engine should draw as the canvas staged preview:
 * the selected finished candidate when visible, otherwise the selected
 * placeholder's live denoise-progress frame, otherwise nothing. Kept React-free
 * so the candidate-vs-progress decision is unit-testable in node.
 */

import type { StagedPreviewInput } from '@workbench/canvas-engine/engine';

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
  /** The current bbox size (document px) — progress frames scale to fill it. */
  bboxWidth: number;
  bboxHeight: number;
}

/**
 * Resolves the staged-preview source. A selected finished candidate wins when
 * candidate previews are visible; otherwise selected-placeholder progress fills
 * the bbox. Returns `null` when neither source is available.
 */
export const selectStagedPreviewSource = ({
  bboxHeight,
  bboxWidth,
  isGenerationInFlight,
  isVisible,
  progressImage,
  selectedImageName,
}: StagedPreviewSelection): StagedPreviewInput | null => {
  if (isVisible && selectedImageName) {
    return { imageName: selectedImageName };
  }
  if (isGenerationInFlight && progressImage && bboxWidth > 0 && bboxHeight > 0) {
    // Progress frames are low-res latents of the bbox region; scale to fill it.
    return { dataUrl: progressImage.dataUrl, height: bboxHeight, width: bboxWidth };
  }
  return null;
};

/**
 * A stable string key for a {@link StagedPreviewInput}, so a React effect only
 * re-drives the (async, decoding) `setStagedPreview` when the source actually
 * changes — including every new progress frame, but not on unrelated renders.
 */
export const stagedPreviewKey = (source: StagedPreviewInput | null): string => {
  if (source === null) {
    return 'none';
  }
  if ('imageName' in source) {
    return `image:${source.imageName}`;
  }
  return `data:${source.width}x${source.height}:${source.dataUrl}`;
};
