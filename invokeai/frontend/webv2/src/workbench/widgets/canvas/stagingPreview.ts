/**
 * Pure selection of what the engine should draw as the canvas staged preview:
 * a live denoise-progress frame while a canvas generation is in flight,
 * otherwise the selected staged candidate, otherwise nothing. Kept React-free
 * so the progress-vs-candidate decision is unit-testable in node.
 */

import type { StagedPreviewInput } from '@workbench/canvas-engine/engine';

/** Inputs to {@link selectStagedPreviewSource}. */
export interface StagedPreviewSelection {
  /** Whether a canvas-destination generation is currently pending/running. */
  isGenerationInFlight: boolean;
  /** The latest denoise-progress frame, if any (b64 data URL + native dims). */
  progressImage: { dataUrl: string; width: number; height: number } | null;
  /** Whether the staging area's preview toggle is on. */
  isVisible: boolean;
  /** The currently selected staged candidate's image name, if any. */
  selectedImageName: string | null;
  /** The current bbox size (document px) — progress frames scale to fill it. */
  bboxWidth: number;
  bboxHeight: number;
}

/**
 * Resolves the staged-preview source. Progress wins while generating (so the
 * canvas shows live denoising over the bbox), handing off to the final
 * candidate once the result lands and the run settles. Returns `null` when the
 * preview is hidden or there is nothing to show.
 */
export const selectStagedPreviewSource = ({
  bboxHeight,
  bboxWidth,
  isGenerationInFlight,
  isVisible,
  progressImage,
  selectedImageName,
}: StagedPreviewSelection): StagedPreviewInput | null => {
  if (isGenerationInFlight && progressImage && bboxWidth > 0 && bboxHeight > 0) {
    // Progress frames are low-res latents of the bbox region; scale to fill it.
    return { dataUrl: progressImage.dataUrl, height: bboxHeight, width: bboxWidth };
  }
  if (isVisible && selectedImageName) {
    return { imageName: selectedImageName };
  }
  return null;
};

/** The app-global latest progress frame, tagged with the queue item it belongs to. */
export interface TargetedProgressImage {
  dataUrl: string;
  width: number;
  height: number;
  target?: { queueItemId: string };
}

/**
 * Gates the app-global latest denoise-progress frame to THIS canvas: returns the
 * frame only when its originating queue item is one of the project's pending/
 * running canvas-destination items. The progress store keeps a single global
 * "latest" frame, so without this filter a generate-widget run in the same
 * project — or a run in another project entirely — would draw its denoise frames
 * over this canvas's bbox. Returning `null` for a foreign frame also lets a
 * landed candidate win over another item's live progress (the candidate is the
 * next fallback in {@link selectStagedPreviewSource}).
 */
export const selectCanvasProgressImage = (
  latest: TargetedProgressImage | null,
  canvasQueueItemIds: ReadonlySet<string>
): { dataUrl: string; width: number; height: number } | null => {
  if (!latest?.target || !canvasQueueItemIds.has(latest.target.queueItemId)) {
    return null;
  }
  return { dataUrl: latest.dataUrl, height: latest.height, width: latest.width };
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
