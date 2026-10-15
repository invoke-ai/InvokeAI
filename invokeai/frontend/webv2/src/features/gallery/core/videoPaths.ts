/**
 * Media cookies authenticate video routes because elements cannot set bearer headers. The full route supports
 * native Range requests for seeking.
 */

export const getGalleryVideoThumbnailPath = (videoName: string): string =>
  `/api/v1/videos/i/${encodeURIComponent(videoName)}/thumbnail`;

export const getGalleryVideoFullPath = (videoName: string): string =>
  `/api/v1/videos/i/${encodeURIComponent(videoName)}/full`;
