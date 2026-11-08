import { getGalleryVideoFullPath, getGalleryVideoThumbnailPath } from '@features/gallery/core/videoPaths';
import { absolutizeApiUrl } from '@platform/transport/http';

export const getGalleryVideoThumbnailUrl = (videoName: string): string =>
  absolutizeApiUrl(getGalleryVideoThumbnailPath(videoName));

export const getGalleryVideoFullUrl = (videoName: string): string =>
  absolutizeApiUrl(getGalleryVideoFullPath(videoName));
