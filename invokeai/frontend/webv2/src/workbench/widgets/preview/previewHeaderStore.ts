import { createExternalStore } from '@workbench/externalStore';

/**
 * Header context published by the preview view so the widget frame's label can
 * show "[board] / [image]" without refetching boards. The preview is a
 * singleton widget (`allowMultiple: false`), so one module-level store is safe.
 * Cleared when the view unmounts or nothing is selected; the label falls back
 * to the static widget title.
 */
export interface PreviewHeaderContext {
  boardName: string | null;
  imageName: string | null;
}

const emptyContext: PreviewHeaderContext = { boardName: null, imageName: null };

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
