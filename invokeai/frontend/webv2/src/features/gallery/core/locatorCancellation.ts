export interface GalleryLocatorRequest {
  signal: AbortSignal;
  release: () => void;
}

const activeGalleryLocatorRequests = new Set<AbortController>();

/** Create a cancellable locator lifetime shared by Gallery reveal entry points. */
export const createGalleryLocatorRequest = (): GalleryLocatorRequest => {
  const controller = new AbortController();
  activeGalleryLocatorRequests.add(controller);

  return {
    signal: controller.signal,
    release: () => activeGalleryLocatorRequests.delete(controller),
  };
};

/** A later Gallery navigation supersedes every locator currently in flight. */
export const abortGalleryLocatorRequests = (): void => {
  for (const controller of activeGalleryLocatorRequests) {
    controller.abort();
  }

  activeGalleryLocatorRequests.clear();
};
