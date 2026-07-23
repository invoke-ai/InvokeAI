/**
 * Regression test for silent video upload failures (PR #9163 review).
 *
 * The bug: uploadVideos() aggregated with Promise.allSettled and dropped rejections, and
 * no uploadVideo.matchRejected listener existed — so a rejected MP4 vanished with no
 * error feedback, and a partially successful batch gave no indication of which files
 * failed. The rejected listener now toasts per failed mutation; these tests pin that the
 * failed file's name reaches the user-visible description.
 */
import { describe, expect, it } from 'vitest';

import { getVideoUploadFailedDescription } from './videoUploadFailedDescription';

describe('getVideoUploadFailedDescription', () => {
  it('names the failed file so mixed-outcome batches are attributable', () => {
    expect(getVideoUploadFailedDescription('clip.mp4', 'Request failed with status code 500')).toBe(
      'clip.mp4: Request failed with status code 500'
    );
  });

  it('still names the file when no error message is available', () => {
    expect(getVideoUploadFailedDescription('clip.mp4', undefined)).toBe('clip.mp4');
  });

  it('falls back to the error message alone for pasted blobs without a name', () => {
    expect(getVideoUploadFailedDescription(undefined, 'Network Error')).toBe('Network Error');
  });

  it('returns undefined when there is nothing to show (toast falls back to its title)', () => {
    expect(getVideoUploadFailedDescription(undefined, undefined)).toBeUndefined();
  });
});
