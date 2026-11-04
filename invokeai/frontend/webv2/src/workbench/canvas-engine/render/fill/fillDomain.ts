// SPDX-License-Identifier: Apache-2.0
/**
 * Where a fill may reach, in document pixels: the generation frame united with the sampled content's bounds, then
 * intersected with the selection's bounds when a selection exists and with the frame when Clip to frame is on. A
 * click on an empty layer fills the frame; a click on content past the frame follows the content.
 */

import type { Rect } from '@workbench/canvas-engine/types';

/** The largest domain a fill samples: one 4 B/px sample plus the worker's level map and region planes. */
export const MAX_FILL_PIXELS = 4096 * 4096;

export interface FillDomainInput {
  readonly frame: Rect;
  /** Document bounds of the content the fill samples; null when it has none. */
  readonly contentBounds: Rect | null;
  /** Document bounds of the active selection; null without a selection. */
  readonly selectionBounds: Rect | null;
  readonly clipToFrame: boolean;
}

interface Edges {
  left: number;
  top: number;
  right: number;
  bottom: number;
}

/** Whole pixels touched by `rect`: fractional edges round outwards. */
const edgesOf = (rect: Rect): Edges => ({
  bottom: Math.ceil(rect.y + rect.height),
  left: Math.floor(rect.x),
  right: Math.ceil(rect.x + rect.width),
  top: Math.floor(rect.y),
});

const intersect = (edges: Edges, rect: Rect): void => {
  const other = edgesOf(rect);
  edges.left = Math.max(edges.left, other.left);
  edges.top = Math.max(edges.top, other.top);
  edges.right = Math.min(edges.right, other.right);
  edges.bottom = Math.min(edges.bottom, other.bottom);
};

/** The integer document rect a fill samples, or null when nothing remains. */
export const resolveFillDomain = ({
  clipToFrame,
  contentBounds,
  frame,
  selectionBounds,
}: FillDomainInput): Rect | null => {
  const edges = edgesOf(frame);
  if (contentBounds && contentBounds.width > 0 && contentBounds.height > 0) {
    const content = edgesOf(contentBounds);
    edges.left = Math.min(edges.left, content.left);
    edges.top = Math.min(edges.top, content.top);
    edges.right = Math.max(edges.right, content.right);
    edges.bottom = Math.max(edges.bottom, content.bottom);
  }
  if (selectionBounds) {
    intersect(edges, selectionBounds);
  }
  if (clipToFrame) {
    intersect(edges, frame);
  }
  const width = edges.right - edges.left;
  const height = edges.bottom - edges.top;
  return width > 0 && height > 0 ? { height, width, x: edges.left, y: edges.top } : null;
};

export const exceedsFillPixelCap = (domain: Rect): boolean => domain.width * domain.height > MAX_FILL_PIXELS;
