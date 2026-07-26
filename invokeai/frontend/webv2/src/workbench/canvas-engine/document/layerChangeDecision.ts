import type { CanvasLayerContract } from '@workbench/canvas-engine/contracts';

import { getLayerThumbnailDisplayKey } from '@workbench/canvas-engine/render/thumbnail';

/** The display token given to a thumbnail whose appearance changed without its pixels. */
const FIRST_DISPLAY_TOKEN = -1;

export interface LayerChangeInput {
  /** The layer as it now stands, or `undefined` when it is no longer in the document. */
  readonly layer: CanvasLayerContract | undefined;
  /** Whether the mirror reported this layer's SOURCE reference as changed. */
  readonly sourceChanged: boolean;
  /** The image this layer referenced before the change, if any. */
  readonly previousImageName: string | null | undefined;
  /** The display key recorded for this layer's last-rendered thumbnail. */
  readonly currentThumbnailKey: string | undefined;
  /** The version recorded for this layer's thumbnail, if any. */
  readonly currentThumbnailVersion: number | undefined;
  /** Whether an open transform session belongs to this layer. */
  readonly hasTransformSession: boolean;
  /** Whether an open text-edit session belongs to this layer. */
  readonly hasTextEditSession: boolean;
  /**
   * Whether this source swap is the bitmap store's own echo. Lazy because it is
   * only ever asked on the source-changed path, and answering costs a lookup.
   */
  readonly isSelfEcho: () => boolean;
}

export type LayerChangeDecision =
  | {
      readonly kind: 'removed';
      readonly releaseImageName: string | null;
      readonly cancelTransformSession: boolean;
      readonly cancelTextEditSession: boolean;
    }
  | {
      readonly kind: 'appearance-only';
      /** The new display key and the token to record, or `null` when it did not change. */
      readonly thumbnailDisplay: { key: string; version: number } | null;
    }
  | {
      readonly kind: 'source-changed';
      readonly thumbnailKey: string;
      readonly releaseImageName: string | null;
      /** `false` when the swap is the bitmap store's own echo, whose pixels the cache already holds. */
      readonly invalidateCache: boolean;
    };

/**
 * What has to happen to one layer's engine-side state after the document
 * changed it. Pure: the caller performs the effects in loop order.
 *
 * The three outcomes are genuinely different, and conflating them has real
 * costs. A prop or transform change — opacity, blend, lock, visibility, rename,
 * nudge — replaces the layer object while leaving its SOURCE reference intact,
 * so the rasterized pixels remain valid. Invalidating there would be wasteful
 * for an image layer and *destructive* for an unflushed paint layer, whose
 * `bitmap: null` source rasterizes to a cleared surface and would wipe strokes
 * that live only in the cache until their debounced upload lands. The
 * compositor applies transform, opacity and blend at draw time, so a recomposite
 * is all such a change needs.
 *
 * A genuine source swap does invalidate — unless it is the echo of a write the
 * bitmap store just made, where the cache already holds exactly those pixels and
 * re-rasterizing would re-fetch them and risk a flicker.
 */
export const decideLayerChange = (input: LayerChangeInput): LayerChangeDecision => {
  const { layer, previousImageName } = input;
  const releaseImageName = previousImageName ?? null;

  if (!layer) {
    return {
      cancelTextEditSession: input.hasTextEditSession,
      cancelTransformSession: input.hasTransformSession,
      kind: 'removed',
      releaseImageName,
    };
  }

  if (!input.sourceChanged) {
    const key = getLayerThumbnailDisplayKey(layer);
    if (key === input.currentThumbnailKey) {
      return { kind: 'appearance-only', thumbnailDisplay: null };
    }
    // Cache versions are positive, so a negative display token can never collide
    // with the next cache publication and suppress its redraw. Each further
    // appearance change steps further negative rather than resetting.
    const current = input.currentThumbnailVersion;
    return {
      kind: 'appearance-only',
      thumbnailDisplay: {
        key,
        version: current !== undefined && current < 0 ? current - 1 : FIRST_DISPLAY_TOKEN,
      },
    };
  }

  return {
    invalidateCache: !input.isSelfEcho(),
    kind: 'source-changed',
    releaseImageName,
    thumbnailKey: getLayerThumbnailDisplayKey(layer),
  };
};
