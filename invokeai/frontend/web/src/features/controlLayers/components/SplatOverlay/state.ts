import { logger } from 'app/logging/logger';
import type { Err, Ok } from 'common/util/result';
import { toast } from 'features/toast/toast';
import { t } from 'i18next';
import { atom } from 'nanostores';

const log = logger('canvas');

export type SplatRect = { x: number; y: number; width: number; height: number };

type SplatOverlayState =
  | { status: 'loading'; sessionId: string; rect: SplatRect }
  | { status: 'ready'; sessionId: string; assetUrl: string; rect: SplatRect };

/**
 * Transient state for the in-canvas 3D (Gaussian-splat) overlay. `null` means the overlay is closed.
 * The `rect` is the overlay's footprint in canvas/world coords — seeded from the source raster layer's
 * bbox, then moved/resized by the user before committing. The bake step renders at this rect.
 */
export const $splatOverlay = atom<SplatOverlayState | null>(null);

/** Move/resize the overlay footprint. No-op if the overlay is closed. */
export const updateSplatOverlayRect = (rect: SplatRect): void => {
  const state = $splatOverlay.get();
  if (!state) {
    return;
  }
  $splatOverlay.set({ ...state, rect });
};

// Tracks the in-flight generation so closing/cancelling aborts the backend run, not just the overlay UI.
let activeGenerationAbort: AbortController | null = null;

export const setSplatGenerationAbort = (controller: AbortController | null): void => {
  activeGenerationAbort = controller;
};

/** Clears the abort slot only if it still holds `controller` — a newer session may have taken it over. */
export const clearSplatGenerationAbort = (controller: AbortController): void => {
  if (activeGenerationAbort === controller) {
    activeGenerationAbort = null;
  }
};

export const clearSplatOverlay = (): void => {
  activeGenerationAbort?.abort();
  activeGenerationAbort = null;
  $splatOverlay.set(null);
};

/**
 * Applies a finished conversion's outcome to the overlay. Every write is gated on the overlay still
 * showing *this* session — the user may have cancelled (state null) or started another conversion
 * (different sessionId) while the run was in flight. A user-initiated cancel/replace also surfaces
 * here as an aborted-run error; the same gate keeps those silent, so the error toast only fires for
 * genuine failures of the session the user is still watching.
 */
export const applyConvertTo3DResult = (result: Ok<string> | Err<Error>, sessionId: string): void => {
  const current = $splatOverlay.get();
  const isCurrentSession = current?.status === 'loading' && current.sessionId === sessionId;

  if (result.isErr()) {
    log.error({ error: String(result.error) }, 'Failed to convert image to 3D');
    if (isCurrentSession) {
      toast({ status: 'error', title: t('controlLayers.convertTo3D.generationError') });
      clearSplatOverlay();
    }
    return;
  }

  if (isCurrentSession) {
    // Use the overlay's *current* rect, not one captured at start — the frame is movable while loading.
    $splatOverlay.set({ status: 'ready', sessionId, assetUrl: result.value, rect: current.rect });
  }
};
