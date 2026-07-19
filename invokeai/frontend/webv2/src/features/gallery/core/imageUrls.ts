import { absolutizeApiUrl } from '@platform/transport/http';

export const getGalleryImageThumbnailUrl = (imageName: string): string =>
  absolutizeApiUrl(`/api/v1/images/i/${encodeURIComponent(imageName)}/thumbnail`);

export const getGalleryImageFullUrl = (imageName: string): string =>
  absolutizeApiUrl(`/api/v1/images/i/${encodeURIComponent(imageName)}/full`);
