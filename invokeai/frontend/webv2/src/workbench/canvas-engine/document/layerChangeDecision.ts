import type { CanvasLayerContract } from '@workbench/canvas-engine/contracts';

import { getLayerThumbnailDisplayKey } from '@workbench/canvas-engine/render/thumbnail';

/** The display token given to a thumbnail whose appearance changed without its pixels. */
const FIRST_DISPLAY_TOKEN = -1;

export interface LayerChangeInput {
  /** The layer as it now stands, or `undefined` when it is no longer in the document. */
  readonly layer: CanvasLayerContract | undefined;
  /** Whether the mirror reported this layer's SOURCE reference as changed. */
  readonly sourceChanged: boolean;
  /** The display key recorded for this layer's last-rendered thumbnail. */
  readonly currentThumbnailKey: string | undefined;
  /** The version recorded for this layer's thumbnail, if any. */
  readonly currentThumbnailVersion: number | undefined;
  /** Whether an open transform session belongs to this layer. */
  readonly hasTransformSession: boolean;
  /** Whether an open text-edit session belongs to this layer. */
  readonly hasTextEditSession: boolean;
  /** Lazily checks bitmap self-echo only when the source changed. */
  readonly isSelfEcho: () => boolean;
}

export type LayerChangeDecision =
  | {
      readonly kind: 'removed';
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
      /** `false` when the swap is the bitmap store's own echo, whose pixels the cache already holds. */
      readonly invalidateCache: boolean;
    };

/**
 * Pure layer-change decisions, applied by the caller in order. Properties and transforms need only recomposition;
 * rerasterizing them could erase unflushed paint. Source swaps invalidate except bitmap-store self-echoes, whose
 * pixels already match.
 */
export const decideLayerChange = (input: LayerChangeInput): LayerChangeDecision => {
  const { layer } = input;

  if (!layer) {
    return {
      cancelTextEditSession: input.hasTextEditSession,
      cancelTransformSession: input.hasTransformSession,
      kind: 'removed',
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
    thumbnailKey: getLayerThumbnailDisplayKey(layer),
  };
};
