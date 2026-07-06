import { atom } from 'nanostores';

export type SplatRect = { x: number; y: number; width: number; height: number };

type SplatOverlayState =
  | { status: 'loading'; rect: SplatRect }
  | { status: 'ready'; assetUrl: string; rect: SplatRect };

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

export const clearSplatOverlay = (): void => {
  activeGenerationAbort?.abort();
  activeGenerationAbort = null;
  $splatOverlay.set(null);
};
