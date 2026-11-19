import type { GalleryItem } from '@features/gallery';
import type { ImageActions, ImageContextMenuVariant } from '@workbench/image-actions';

import { registerAccountOwnedResource } from '@platform/state/accountLifecycle';
import { createExternalStore } from '@platform/state/externalStore';

import type { PreviewDensity } from './previewDensity';

/** The loupe's commands, published once per selection; the readout lives in `previewStageStore`. */
export interface PreviewZoomCommands {
  reset: () => void;
  /** Zoom to a fraction of the image's own pixels (1 = 100%), within `getActualZoomLimits`. */
  zoomTo: (actualZoom: number) => void;
}

export type PreviewZoomControls = PreviewZoomCommands & PreviewZoomReadout;

/** What the loupe shows, as percents of the image's own pixels. */
export interface PreviewZoomReadout {
  /** What the fitted image shows, unrounded, for the menu's presets. */
  fitPercent: number | null;
  isZoomed: boolean;
  /** What is on screen right now. */
  percent: number | null;
}

/** Max zoom, as a fraction of the image's actual pixel size. */
const MAX_ACTUAL_ZOOM = 8;

/** Zoom range in image pixels: from min(fit, 100%) up to max(`MAX_ACTUAL_ZOOM`, fit). */
export const getActualZoomLimits = (fitRatio: number): { max: number; min: number } => ({
  max: Math.max(fitRatio, MAX_ACTUAL_ZOOM),
  min: Math.min(fitRatio, 1),
});

/** Where the selection sits in its board, for the Details panel's header line. */
export interface PreviewItemPosition {
  boardItemCount: number;
  isLoadingBoard: boolean;
  selectedIndex: number;
}

/** Singleton Preview selection for external chrome; cleared on unmount or empty selection. */
export interface PreviewHeaderContext {
  /** The selected item with board/star context, ready for common actions. */
  actionItem: GalleryItem | null;
  /** The view's `useImageActions` instance (carries delete-neighbor handling). */
  actions: ImageActions | null;
  boardName: string | null;
  /** Set only while Select for compare waits for a second image: switches it off again. */
  cancelCompare: (() => void) | null;
  copyCurrentVideoFrame: (() => void) | null;
  /** Preview's focusable region, where a header control removed while it holds focus hands it on. */
  getRegion: (() => HTMLElement | null) | null;
  isVideoFrameCopyAvailable: boolean;
  itemName: string | null;
  /** Opens the shared menu at viewport coordinates; `housekeeping` omits Edit and Use items. */
  openItemMenu: ((x: number, y: number, variant: ImageContextMenuVariant) => void) | null;
  position: PreviewItemPosition | null;
  /** Null unless the selection is an image the loupe can zoom. */
  zoom: PreviewZoomCommands | null;
}

const emptyContext: PreviewHeaderContext = {
  actionItem: null,
  actions: null,
  boardName: null,
  cancelCompare: null,
  copyCurrentVideoFrame: null,
  getRegion: null,
  isVideoFrameCopyAvailable: false,
  itemName: null,
  openItemMenu: null,
  position: null,
  zoom: null,
};

const store = createExternalStore<PreviewHeaderContext>(emptyContext);

export const previewHeaderStore = {
  clear(): void {
    store.patchSnapshot(emptyContext);
  },
  set(context: PreviewHeaderContext): void {
    store.patchSnapshot(context);
  },
};

registerAccountOwnedResource({
  clear: previewHeaderStore.clear,
  name: 'preview-header',
});

export const usePreviewHeaderContext = (): PreviewHeaderContext => store.useSelector((snapshot) => snapshot);

/** Publish hot-path zoom readouts directly so wheel ticks update only the zoom menu. */
export interface PreviewStageContext {
  /** The view's measured density, so the header sizes its controls by width; null until the view has measured. */
  density: PreviewDensity | null;
  /** The media stage; the Details panel is pinned inside it, off the filmstrip. */
  stageElement: HTMLElement | null;
  zoom: PreviewZoomReadout | null;
}

const emptyStage: PreviewStageContext = { density: null, stageElement: null, zoom: null };
const stageStore = createExternalStore<PreviewStageContext>(emptyStage);

export const previewStageStore = {
  clear(): void {
    stageStore.patchSnapshot(emptyStage);
  },
  setDensity(density: PreviewDensity | null): void {
    stageStore.patchSnapshot({ density });
  },
  setStageElement(stageElement: HTMLElement | null): void {
    stageStore.patchSnapshot({ stageElement });
  },
  setZoom(zoom: PreviewZoomReadout | null): void {
    stageStore.patchSnapshot({ zoom });
  },
};

registerAccountOwnedResource({
  clear: previewStageStore.clear,
  name: 'preview-stage',
});

export const usePreviewStageContext = (): PreviewStageContext => stageStore.useSelector((snapshot) => snapshot);

/** Only the measured density, for readers that must not re-render on every zoom readout. */
export const usePreviewStageDensity = (): PreviewDensity | null =>
  stageStore.useSelector((snapshot) => snapshot.density);
