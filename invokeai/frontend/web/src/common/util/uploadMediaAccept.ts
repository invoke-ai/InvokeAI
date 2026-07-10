import type { Accept } from 'react-dropzone';

/**
 * Single source of truth for which media files the client accepts for upload.
 *
 * The video lists must stay in sync with the video upload router
 * (`invokeai/app/api/routers/videos.py`: `ACCEPTED_VIDEO_MIME_PREFIXES` /
 * `ACCEPTED_VIDEO_EXTENSIONS`). The backend accepts MP4 only — it stores uploads under a
 * `.mp4` name without transcoding — so advertising other containers client-side just
 * produces a guaranteed 415 after the bytes have been uploaded.
 */
export const ACCEPTED_IMAGE_TYPES = ['image/png', 'image/jpg', 'image/jpeg', 'image/webp'];
export const ACCEPTED_IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.webp'];
export const ACCEPTED_VIDEO_TYPES = ['video/mp4'];
export const ACCEPTED_VIDEO_EXTENSIONS = ['.mp4'];

const addUpperCaseReducer = (acc: string[], ext: string) => {
  acc.push(ext);
  acc.push(ext.toUpperCase());
  return acc;
};

/** react-dropzone accept map for image-only upload fields (board covers, style presets, etc). */
export const imageDropzoneAccept: Accept = {
  'image/png': ['.png'].reduce(addUpperCaseReducer, [] as string[]),
  'image/jpeg': ['.jpg', '.jpeg', '.png'].reduce(addUpperCaseReducer, [] as string[]),
  'image/webp': ['.webp'].reduce(addUpperCaseReducer, [] as string[]),
};

/** react-dropzone accept map for the gallery uploader, which also takes videos. */
export const imageAndVideoDropzoneAccept: Accept = {
  ...imageDropzoneAccept,
  'video/mp4': ['.mp4'].reduce(addUpperCaseReducer, [] as string[]),
};

/** Returns true when the file is an uploadable video (by MIME or by extension). */
export const isVideoFile = (file: File): boolean => {
  if (file.type && ACCEPTED_VIDEO_TYPES.includes(file.type.toLowerCase())) {
    return true;
  }
  const lower = file.name.toLowerCase();
  return ACCEPTED_VIDEO_EXTENSIONS.some((ext) => lower.endsWith(ext));
};
