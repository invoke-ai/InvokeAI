import { getGalleryImageFullPath, getGalleryImageThumbnailPath } from '@features/gallery/core/imagePaths';
import { absolutizeApiUrl } from '@platform/transport/http';

export const getGalleryImageThumbnailUrl = (imageName: string): string =>
  absolutizeApiUrl(getGalleryImageThumbnailPath(imageName));

export const getGalleryImageFullUrl = (imageName: string): string =>
  absolutizeApiUrl(getGalleryImageFullPath(imageName));
