import type { AccountScope } from '@platform/state/accountLifecycle';

import { isAccountScopeCurrent } from '@platform/state/accountLifecycle';

let revision = 0;
const listeners = new Set<() => void>();

export const getGalleryThumbnailRevision = (): number => revision;

export const subscribeGalleryThumbnailRevision = (listener: () => void): (() => void) => {
  listeners.add(listener);
  return () => listeners.delete(listener);
};

/** Ask mounted gallery thumbnails to request their URLs again after maintenance repairs them. */
export const refreshGalleryThumbnails = (owner: AccountScope): void => {
  if (!isAccountScopeCurrent(owner)) {
    return;
  }

  revision += 1;
  listeners.forEach((listener) => listener());
};

export const getRefreshedGalleryThumbnailUrl = (url: string, currentRevision: number): string => {
  if (currentRevision === 0 || /^(?:blob|data):/i.test(url)) {
    return url;
  }

  const refreshedUrl = new URL(url, window.location.href);
  refreshedUrl.searchParams.set('gallery_thumbnail_revision', String(currentRevision));
  return refreshedUrl.toString();
};
