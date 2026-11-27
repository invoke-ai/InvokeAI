/** Media cookies authenticate video routes (elements cannot set bearer headers); the full route serves Range seeks. */

export const getGalleryVideoThumbnailPath = (videoName: string): string =>
  `/api/v1/videos/i/${encodeURIComponent(videoName)}/thumbnail`;

export const getGalleryVideoFullPath = (videoName: string): string =>
  `/api/v1/videos/i/${encodeURIComponent(videoName)}/full`;
