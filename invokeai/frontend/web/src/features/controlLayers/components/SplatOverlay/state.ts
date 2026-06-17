import { atom } from 'nanostores';

type SplatRect = { x: number; y: number; width: number; height: number };

type SplatOverlayState =
  | { status: 'loading'; rect: SplatRect }
  | { status: 'ready'; assetUrl: string; rect: SplatRect };

/**
 * Transient state for the in-canvas 3D (Gaussian-splat) overlay. `null` means the overlay is closed.
 * The `rect` is the source raster layer's bbox in canvas/world coords (used by the bake step).
 */
export const $splatOverlay = atom<SplatOverlayState | null>(null);

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
