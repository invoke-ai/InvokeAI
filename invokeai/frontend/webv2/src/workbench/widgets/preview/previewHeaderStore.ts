import type { GalleryItem } from '@features/gallery';
import type { ImageActions } from '@workbench/image-actions';

import { registerAccountOwnedResource } from '@platform/state/accountLifecycle';
import { createExternalStore } from '@platform/state/externalStore';

/** The loupe's commands, published once per selection; the readout lives in `previewStageStore`. */
export interface PreviewZoomCommands {
  reset: () => void;
  /** Zoom to a fraction of the image's own pixels (1 = 100%), never below fit. */
  zoomTo: (actualZoom: number) => void;
}

/** What the header's zoom menu renders: the readout plus the commands. */
export type PreviewZoomControls = PreviewZoomCommands & PreviewZoomReadout;

/** What the loupe shows, as percents of the image's own pixels. */
export interface PreviewZoomReadout {
  /** What the fitted image shows, for the menu's presets. */
  fitPercent: number | null;
  isZoomed: boolean;
  /** What is on screen right now. */
  percent: number | null;
}

/** Where the selection sits in its board, for the Details popover's header line. */
export interface PreviewItemPosition {
  boardItemCount: number;
  isLoadingBoard: boolean;
  selectedIndex: number;
}

/**
 * Publish singleton Preview selection/actions for external chrome without refetching. Clear on unmount or empty
 * selection so static chrome returns.
 */
export interface PreviewHeaderContext {
  /** The selected item with board/star context, ready for common actions. */
  actionItem: GalleryItem | null;
  /** The view's `useImageActions` instance (carries delete-neighbor handling). */
  actions: ImageActions | null;
  boardName: string | null;
  copyCurrentVideoFrame: (() => void) | null;
  isVideoFrameCopyAvailable: boolean;
  itemName: string | null;
  /** Open the view's shared image context menu from header actions at viewport coordinates. */
  openItemMenu: ((x: number, y: number) => void) | null;
  position: PreviewItemPosition | null;
  /** Null unless the selection is an image the loupe can zoom. */
  zoom: PreviewZoomCommands | null;
}

const emptyContext: PreviewHeaderContext = {
  actionItem: null,
  actions: null,
  boardName: null,
  copyCurrentVideoFrame: null,
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
  /** The media stage; the Details popover keeps within it, off the filmstrip. */
  stageElement: HTMLElement | null;
  zoom: PreviewZoomReadout | null;
}

const emptyStage: PreviewStageContext = { stageElement: null, zoom: null };
const stageStore = createExternalStore<PreviewStageContext>(emptyStage);

export const previewStageStore = {
  clear(): void {
    stageStore.patchSnapshot(emptyStage);
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
