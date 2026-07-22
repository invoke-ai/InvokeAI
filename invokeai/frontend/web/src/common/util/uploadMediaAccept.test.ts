/**
 * Regression tests for client-side upload acceptance (PR #9163 review).
 *
 * The bug: the dropzone accept map advertised `.webm`/`.mov` and `isVideoFile` treated
 * `.webm`/`.mov`/`.mkv` as uploadable videos, but the video upload router accepts MP4 only
 * (`invokeai/app/api/routers/videos.py`: `ACCEPTED_VIDEO_MIME_PREFIXES = ("video/mp4",)`,
 * `ACCEPTED_VIDEO_EXTENSIONS = (".mp4",)`). Dropping a WebM or QuickTime file started an
 * upload the server was guaranteed to reject with 415.
 *
 * These tests pin the client lists to the backend contract: if a new container is ever
 * accepted server-side, update `uploadMediaAccept.ts` and these expectations together.
 */
import { describe, expect, it } from 'vitest';

import {
  ACCEPTED_VIDEO_EXTENSIONS,
  ACCEPTED_VIDEO_TYPES,
  getUploadDropzoneAccept,
  imageAndVideoDropzoneAccept,
  imageDropzoneAccept,
  isAcceptedUploadFile,
  isImageFile,
  isVideoFile,
  partitionUploadFiles,
} from './uploadMediaAccept';

// `isVideoFile` only reads `type` and `name`, so a plain object stands in for a DOM File.
const fakeFile = (name: string, type: string): File => ({ name, type }) as File;

describe('video acceptance matches the backend (MP4 only)', () => {
  it('accepts only MIME types and extensions the video upload router supports', () => {
    expect(ACCEPTED_VIDEO_TYPES).toEqual(['video/mp4']);
    expect(ACCEPTED_VIDEO_EXTENSIONS).toEqual(['.mp4']);
  });

  it('advertises only backend-supported video entries in the gallery dropzone accept map', () => {
    const videoEntries = Object.entries(imageAndVideoDropzoneAccept).filter(([mime]) => mime.startsWith('video/'));
    expect(videoEntries).toEqual([['video/mp4', ['.mp4', '.MP4']]]);
  });

  it('does not advertise any video entries for image-only upload fields', () => {
    expect(Object.keys(imageDropzoneAccept).some((mime) => mime.startsWith('video/'))).toBe(false);
  });
});

describe('isVideoFile', () => {
  it('recognizes MP4 by MIME type and by extension', () => {
    expect(isVideoFile(fakeFile('clip.mp4', 'video/mp4'))).toBe(true);
    expect(isVideoFile(fakeFile('CLIP.MP4', ''))).toBe(true);
  });

  it('rejects video containers the backend does not accept', () => {
    expect(isVideoFile(fakeFile('clip.webm', 'video/webm'))).toBe(false);
    expect(isVideoFile(fakeFile('clip.mov', 'video/quicktime'))).toBe(false);
    expect(isVideoFile(fakeFile('clip.mkv', 'video/x-matroska'))).toBe(false);
  });

  it('rejects images', () => {
    expect(isVideoFile(fakeFile('image.png', 'image/png'))).toBe(false);
  });
});

describe('isImageFile / isAcceptedUploadFile', () => {
  it('accepts a file when either the MIME type or the extension is recognized', () => {
    // Browsers sometimes supply an empty File.type — the extension must suffice on its own.
    expect(isAcceptedUploadFile(fakeFile('clip.mp4', ''))).toBe(true);
    expect(isAcceptedUploadFile(fakeFile('image.png', ''))).toBe(true);
    // And a recognized MIME type must suffice without a matching extension.
    expect(isAcceptedUploadFile(fakeFile('pasted-blob', 'image/png'))).toBe(true);
    expect(isAcceptedUploadFile(fakeFile('pasted-blob', 'video/mp4'))).toBe(true);
  });

  it('rejects unsupported media', () => {
    expect(isImageFile(fakeFile('vector.svg', 'image/svg+xml'))).toBe(false);
    expect(isAcceptedUploadFile(fakeFile('clip.webm', 'video/webm'))).toBe(false);
    expect(isAcceptedUploadFile(fakeFile('notes.txt', 'text/plain'))).toBe(false);
  });
});

describe('video uploads are opt-in per consumer', () => {
  const mp4 = fakeFile('clip.mp4', 'video/mp4');
  const png = fakeFile('image.png', 'image/png');

  it('image-only consumers do not advertise video MIME types', () => {
    expect(Object.keys(getUploadDropzoneAccept(false)).some((mime) => mime.startsWith('video/'))).toBe(false);
    expect(getUploadDropzoneAccept(true)).toBe(imageAndVideoDropzoneAccept);
  });

  it('rejects an MP4 submitted to an image-only consumer instead of uploading it', () => {
    const { imageFiles, videoFiles, rejectedFiles } = partitionUploadFiles([png, mp4], false);
    expect(imageFiles).toEqual([png]);
    expect(videoFiles).toEqual([]); // nothing to hand to the video uploader
    expect(rejectedFiles).toEqual([mp4]);
  });

  it('routes an MP4 to the video batch when the consumer opted in', () => {
    const { imageFiles, videoFiles, rejectedFiles } = partitionUploadFiles([png, mp4], true);
    expect(imageFiles).toEqual([png]);
    expect(videoFiles).toEqual([mp4]);
    expect(rejectedFiles).toEqual([]);
  });
});
