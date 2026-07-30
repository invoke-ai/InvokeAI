/**
 * Backend-relative video media paths, mirroring {@link ./imagePaths}.
 *
 * Both routes authenticate with the path-scoped media cookie rather than a bearer header,
 * because `<video>` and `<img>` cannot set request headers. The full route additionally
 * serves HTTP Range requests, which the `<video>` element issues on its own for seeking.
 */

export const getGalleryVideoThumbnailPath = (videoName: string): string =>
  `/api/v1/videos/i/${encodeURIComponent(videoName)}/thumbnail`;

export const getGalleryVideoFullPath = (videoName: string): string =>
  `/api/v1/videos/i/${encodeURIComponent(videoName)}/full`;
