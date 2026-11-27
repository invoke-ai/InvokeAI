export const getGalleryImageThumbnailPath = (imageName: string): string =>
  `/api/v1/images/i/${encodeURIComponent(imageName)}/thumbnail`;

export const getGalleryImageFullPath = (imageName: string): string =>
  `/api/v1/images/i/${encodeURIComponent(imageName)}/full`;
