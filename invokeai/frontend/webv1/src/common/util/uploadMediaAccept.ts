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
const ACCEPTED_IMAGE_TYPES = ['image/png', 'image/jpg', 'image/jpeg', 'image/webp'];
const ACCEPTED_IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.webp'];
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

/** Returns true when the file is an uploadable image (by MIME or by extension). */
export const isImageFile = (file: File): boolean => {
  if (file.type && ACCEPTED_IMAGE_TYPES.includes(file.type.toLowerCase())) {
    return true;
  }
  const lower = file.name.toLowerCase();
  return ACCEPTED_IMAGE_EXTENSIONS.some((ext) => lower.endsWith(ext));
};

/**
 * Returns true when the file is uploadable media of either kind. MIME and extension each
 * suffice on their own: browsers sometimes supply an empty or generic `File.type` (e.g. for
 * a pasted or drag-dropped MP4), and the backend upload routes likewise accept either signal.
 */
export const isAcceptedUploadFile = (file: File): boolean => isImageFile(file) || isVideoFile(file);

/** The dropzone accept map for an uploader: videos are opt-in, images always accepted. */
export const getUploadDropzoneAccept = (allowVideos: boolean): Accept =>
  allowVideos ? imageAndVideoDropzoneAccept : imageDropzoneAccept;

type UploadPartition = {
  imageFiles: File[];
  videoFiles: File[];
  /** Videos submitted to an uploader that did not opt into videos. */
  rejectedFiles: File[];
};

/**
 * Splits dropped files into per-media upload batches. Videos are only uploadable when the
 * consumer explicitly opted in — an image-only consumer (e.g. a board cover or ref-image
 * field) must reject an MP4 outright rather than upload it to the gallery and silently
 * discard the resulting VideoDTO.
 */
export const partitionUploadFiles = (files: File[], allowVideos: boolean): UploadPartition => {
  const imageFiles: File[] = [];
  const videoFiles: File[] = [];
  const rejectedFiles: File[] = [];
  for (const file of files) {
    if (!isVideoFile(file)) {
      imageFiles.push(file);
    } else if (allowVideos) {
      videoFiles.push(file);
    } else {
      rejectedFiles.push(file);
    }
  }
  return { imageFiles, videoFiles, rejectedFiles };
};
