import type { GalleryImage } from '@features/gallery';

import { registerAccountOwnedResource } from '@platform/state/accountLifecycle';
import { createExternalStore } from '@platform/state/externalStore';

/**
 * The image a new Canvas project opens on, staged by the session controller for the Canvas widget to lay down once
 * it mounts. The controller sizes the canvas itself; placing the image needs the gallery import path, which lives
 * with the widget rather than in the editor's boot graph. Keyed by project so a request never lands elsewhere.
 */
export interface CanvasStartImageRequest {
  image: GalleryImage;
  projectId: string;
}

const store = createExternalStore<{ request: CanvasStartImageRequest | null }>({ request: null });

registerAccountOwnedResource({
  clear: () => store.setSnapshot({ request: null }),
  name: 'canvas-start-image',
});

export const requestCanvasStartImage = (request: CanvasStartImageRequest): void => {
  store.setSnapshot({ request });
};

/** Returns and clears the request staged for `projectId`; null when none, or when it belongs to another project. */
export const takeCanvasStartImage = (projectId: string): GalleryImage | null => {
  const { request } = store.getSnapshot();

  if (request === null || request.projectId !== projectId) {
    return null;
  }

  store.setSnapshot({ request: null });

  return request.image;
};

export const subscribeCanvasStartImage = store.subscribe;
