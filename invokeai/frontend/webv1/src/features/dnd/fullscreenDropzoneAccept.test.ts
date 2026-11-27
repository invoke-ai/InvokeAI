/**
 * Regression tests for the fullscreen drag-drop / paste validator (PR #9163 review).
 *
 * The bug: the validator required BOTH an accepted MIME type AND an accepted extension, so a
 * valid `clip.mp4` whose browser supplied an empty or generic `File.type` was rejected before
 * upload even though the backend accepts MP4 by extension alone.
 */
import { describe, expect, it } from 'vitest';

import { zUploadFile } from './fullscreenDropzoneAccept';

// The validator only reads `type` and `name`, so a plain object stands in for a DOM File.
const fakeFile = (name: string, type: string): File => ({ name, type }) as File;

describe('zUploadFile', () => {
  it('accepts an MP4 with an empty MIME type by its extension', () => {
    expect(zUploadFile.safeParse(fakeFile('clip.mp4', '')).success).toBe(true);
  });

  it('accepts supported media by MIME type alone', () => {
    expect(zUploadFile.safeParse(fakeFile('pasted-blob', 'image/png')).success).toBe(true);
    expect(zUploadFile.safeParse(fakeFile('pasted-blob', 'video/mp4')).success).toBe(true);
  });

  it('rejects video containers the backend does not accept', () => {
    expect(zUploadFile.safeParse(fakeFile('clip.webm', 'video/webm')).success).toBe(false);
    expect(zUploadFile.safeParse(fakeFile('clip.mov', 'video/quicktime')).success).toBe(false);
    expect(zUploadFile.safeParse(fakeFile('clip.mkv', 'video/x-matroska')).success).toBe(false);
  });

  it('rejects non-media files', () => {
    expect(zUploadFile.safeParse(fakeFile('notes.txt', 'text/plain')).success).toBe(false);
  });
});
