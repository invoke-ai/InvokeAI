import type { GalleryImage } from '@workbench/gallery/api';
import type { ImageActions } from '@workbench/image-actions';

import { createExternalStore } from '@workbench/externalStore';

/**
 * Header context published by the preview view so the widget frame's chrome
 * (label + header actions) can render the current selection without refetching
 * boards or re-instantiating image actions. The preview is a singleton widget
 * (`allowMultiple: false`), so one module-level store is safe. Cleared when
 * the view unmounts or nothing is selected; the chrome falls back to the
 * static title and hides the action strip.
 */
export interface PreviewHeaderContext {
  /** The selected image with board/star context, ready for image actions. */
  actionImage: GalleryImage | null;
  /** The view's `useImageActions` instance (carries delete-neighbor handling). */
  actions: ImageActions | null;
  boardName: string | null;
  imageName: string | null;
  /**
   * Opens the view's full image context menu anchored at viewport coordinates.
   * The header's "image actions" dropdown reuses the exact right-click menu —
   * one source of truth for every image verb.
   */
  openImageMenu: ((x: number, y: number) => void) | null;
}

const emptyContext: PreviewHeaderContext = {
  actionImage: null,
  actions: null,
  boardName: null,
  imageName: null,
  openImageMenu: null,
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

export const usePreviewHeaderContext = (): PreviewHeaderContext => store.useSelector((snapshot) => snapshot);
