import type { GalleryItem } from '@features/gallery';
import type { ImageActions } from '@workbench/image-actions';

import { registerAccountOwnedResource } from '@platform/state/accountLifecycle';
import { createExternalStore } from '@platform/state/externalStore';

/**
 * Header context published by the preview view so the widget frame's chrome
 * (label + header actions) can render the current selection without refetching
 * boards or re-instantiating image actions. The preview is a singleton widget
 * (`allowMultiple: false`), so one module-level store is safe. Cleared when
 * the view unmounts or nothing is selected; the chrome falls back to the
 * static title and hides the action strip.
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
  /**
   * Opens the view's full image context menu anchored at viewport coordinates.
   * The header's "image actions" dropdown reuses the exact right-click menu —
   * one source of truth for every image verb.
   */
  openItemMenu: ((x: number, y: number) => void) | null;
  openVideoDetails: (() => void) | null;
}

const emptyContext: PreviewHeaderContext = {
  actionItem: null,
  actions: null,
  boardName: null,
  copyCurrentVideoFrame: null,
  isVideoFrameCopyAvailable: false,
  itemName: null,
  openItemMenu: null,
  openVideoDetails: null,
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
