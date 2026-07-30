import type { GalleryItemRef } from '@features/gallery/core/items';

import { galleryItemOrganization } from '@features/gallery/index';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const MockApiError = vi.hoisted(
  () =>
    class ApiError extends Error {
      readonly status: number;

      constructor(message: string, status: number) {
        super(message);
        this.status = status;
      }
    }
);

const mocks = vi.hoisted(() => ({
  apiFetchJson: vi.fn(),
}));

vi.mock('@platform/transport/http', () => ({
  absolutizeApiUrl: (url: string) => `https://api.test${url}`,
  ApiError: MockApiError,
  apiFetch: vi.fn(),
  apiFetchJson: mocks.apiFetchJson,
  apiFetchRaw: vi.fn(),
  sleep: vi.fn(),
}));

const imageRef = (name: string): GalleryItemRef => ({ kind: 'image', name });
const videoRef = (name: string): GalleryItemRef => ({ kind: 'video', name });

describe('galleryItemOrganization transport and confirmed outcomes', () => {
  beforeEach(() => {
    mocks.apiFetchJson.mockReset();
  });

  it('stars an image-only request and returns its affected board', async () => {
    const controller = new AbortController();
    mocks.apiFetchJson.mockResolvedValue({
      affected_boards: ['board-image'],
      starred_images: ['still.png'],
    });

    await expect(galleryItemOrganization.setStarred([imageRef('still.png')], true, controller.signal)).resolves.toEqual(
      {
        affectedBoardIds: ['board-image'],
        failed: [],
        succeeded: [imageRef('still.png')],
      }
    );
    expect(mocks.apiFetchJson).toHaveBeenCalledWith('/api/v1/images/star', {
      body: JSON.stringify({ image_names: ['still.png'] }),
      method: 'POST',
      signal: controller.signal,
    });
  });

  it('uses the video bulk star, unstar, and delete DTOs without inferring success', async () => {
    mocks.apiFetchJson
      .mockResolvedValueOnce({ affected_boards: ['board-a'], starred_videos: ['clip.mp4'] })
      .mockResolvedValueOnce({ affected_boards: ['board-a'], unstarred_videos: ['clip.mp4'] })
      .mockResolvedValueOnce({
        affected_boards: ['board-a'],
        deleted_videos: ['clip.mp4'],
        failed_videos: [],
      });

    await expect(galleryItemOrganization.setStarred([videoRef('clip.mp4')], true)).resolves.toMatchObject({
      failed: [],
      succeeded: [videoRef('clip.mp4')],
    });
    await expect(galleryItemOrganization.setStarred([videoRef('clip.mp4')], false)).resolves.toMatchObject({
      failed: [],
      succeeded: [videoRef('clip.mp4')],
    });
    await expect(galleryItemOrganization.delete([videoRef('clip.mp4')])).resolves.toEqual({
      affectedBoardIds: ['board-a'],
      failed: [],
      succeeded: [videoRef('clip.mp4')],
    });

    expect(mocks.apiFetchJson).toHaveBeenNthCalledWith(1, '/api/v1/videos/star', {
      body: JSON.stringify({ video_names: ['clip.mp4'] }),
      method: 'POST',
      signal: undefined,
    });
    expect(mocks.apiFetchJson).toHaveBeenNthCalledWith(2, '/api/v1/videos/unstar', {
      body: JSON.stringify({ video_names: ['clip.mp4'] }),
      method: 'POST',
      signal: undefined,
    });
    expect(mocks.apiFetchJson).toHaveBeenNthCalledWith(3, '/api/v1/videos/delete', {
      body: JSON.stringify({ video_names: ['clip.mp4'] }),
      method: 'POST',
      signal: undefined,
    });
  });

  it('isolates same-name media, deduplicates keys, intersects successes, and preserves request order', async () => {
    const refs = [
      videoRef('shared'),
      imageRef('locked.png'),
      imageRef('shared'),
      videoRef('later.mp4'),
      videoRef('shared'),
      imageRef('shared'),
    ];
    mocks.apiFetchJson.mockImplementation((url: string) => {
      if (url === '/api/v1/images/star') {
        return Promise.resolve({
          affected_boards: ['none', 'board-image', 'none'],
          starred_images: ['unexpected.png', 'shared'],
        });
      }

      return Promise.resolve({
        affected_boards: ['board-video', 'none'],
        starred_videos: ['later.mp4', 'unexpected.mp4', 'shared'],
      });
    });

    await expect(galleryItemOrganization.setStarred(refs, true)).resolves.toEqual({
      affectedBoardIds: ['board-image', 'board-video', 'none'],
      failed: [imageRef('locked.png')],
      succeeded: [videoRef('shared'), imageRef('shared'), videoRef('later.mp4')],
    });
    expect(mocks.apiFetchJson).toHaveBeenCalledTimes(2);
    expect(JSON.parse(String(mocks.apiFetchJson.mock.calls[0]?.[1]?.body))).toEqual({
      image_names: ['locked.png', 'shared'],
    });
    expect(JSON.parse(String(mocks.apiFetchJson.mock.calls[1]?.[1]?.body))).toEqual({
      video_names: ['shared', 'later.mp4'],
    });
  });

  it('keeps a fulfilled video delete partition when the image partition rejects', async () => {
    mocks.apiFetchJson.mockImplementation((url: string) => {
      if (url === '/api/v1/images/delete') {
        return Promise.reject(new TypeError('image network failed'));
      }

      return Promise.resolve({
        affected_boards: ['board-video'],
        deleted_videos: ['gone.mp4'],
        failed_videos: ['locked.mp4'],
      });
    });

    await expect(
      galleryItemOrganization.delete([imageRef('gone.png'), videoRef('gone.mp4'), videoRef('locked.mp4')])
    ).resolves.toEqual({
      affectedBoardIds: ['board-video'],
      failed: [imageRef('gone.png'), videoRef('locked.mp4')],
      succeeded: [videoRef('gone.mp4')],
    });
    expect(mocks.apiFetchJson).toHaveBeenCalledTimes(2);
  });

  it('moves mixed media to a real board once per deduplicated item and ignores unconfirmed response effects', async () => {
    const refs = [imageRef('still.png'), videoRef('one.mp4'), videoRef('one.mp4'), videoRef('locked.mp4')];
    mocks.apiFetchJson.mockImplementation((url: string, init?: RequestInit) => {
      const body = JSON.parse(String(init?.body)) as { image_names?: string[]; video_name?: string };

      if (url === '/api/v1/board_images/batch') {
        return Promise.resolve({
          added_images: ['still.png'],
          affected_boards: ['source-image', 'destination'],
        });
      }
      if (body.video_name === 'one.mp4') {
        return Promise.resolve({
          added_videos: ['one.mp4'],
          affected_boards: ['source-video', 'destination'],
        });
      }

      return Promise.resolve({
        added_videos: ['unexpected.mp4'],
        affected_boards: ['must-not-be-reported'],
      });
    });

    await expect(galleryItemOrganization.moveToBoard(refs, 'destination')).resolves.toEqual({
      affectedBoardIds: ['destination', 'source-image', 'source-video'],
      failed: [videoRef('locked.mp4')],
      succeeded: [imageRef('still.png'), videoRef('one.mp4')],
    });
    expect(mocks.apiFetchJson).toHaveBeenCalledTimes(3);
    expect(mocks.apiFetchJson).toHaveBeenCalledWith('/api/v1/board_images/batch', {
      body: JSON.stringify({ board_id: 'destination', image_names: ['still.png'] }),
      method: 'POST',
      signal: undefined,
    });
    expect(mocks.apiFetchJson).toHaveBeenCalledWith('/api/v1/videos/board', {
      body: JSON.stringify({ board_id: 'destination', video_name: 'one.mp4' }),
      method: 'POST',
      signal: undefined,
    });
  });

  it('removes images and videos to none with the distinct remove endpoints', async () => {
    mocks.apiFetchJson.mockImplementation((url: string) => {
      if (url === '/api/v1/board_images/batch/delete') {
        return Promise.resolve({
          affected_boards: ['source-image', 'none'],
          removed_images: ['still.png'],
        });
      }

      return Promise.resolve({
        affected_boards: ['none', 'source-video'],
        removed_videos: ['clip.mp4'],
      });
    });

    await expect(
      galleryItemOrganization.moveToBoard([videoRef('clip.mp4'), imageRef('still.png')], 'none')
    ).resolves.toEqual({
      affectedBoardIds: ['none', 'source-image', 'source-video'],
      failed: [],
      succeeded: [videoRef('clip.mp4'), imageRef('still.png')],
    });
    expect(mocks.apiFetchJson).toHaveBeenCalledWith('/api/v1/board_images/batch/delete', {
      body: JSON.stringify({ image_names: ['still.png'] }),
      method: 'POST',
      signal: undefined,
    });
    expect(mocks.apiFetchJson).toHaveBeenCalledWith('/api/v1/videos/board', {
      body: JSON.stringify({ video_name: 'clip.mp4' }),
      method: 'DELETE',
      signal: undefined,
    });
  });

  it('limits video board movement to four requests in flight', async () => {
    const refs = Array.from({ length: 9 }, (_, index) => videoRef(`clip-${index}.mp4`));
    const releases: (() => void)[] = [];
    let active = 0;
    let maxActive = 0;
    let started = 0;

    mocks.apiFetchJson.mockImplementation((_url: string, init?: RequestInit) => {
      const { video_name: videoName } = JSON.parse(String(init?.body)) as { video_name: string };
      active += 1;
      started += 1;
      maxActive = Math.max(maxActive, active);

      return new Promise((resolve) => {
        releases.push(() => {
          active -= 1;
          resolve({
            added_videos: [videoName],
            affected_boards: ['destination'],
          });
        });
      });
    });

    const movement = galleryItemOrganization.moveToBoard(refs, 'destination');

    while (started < refs.length) {
      await vi.waitFor(() => {
        expect(releases.length).toBeGreaterThan(0);
      });
      releases.shift()?.();
      await Promise.resolve();
    }
    while (releases.length > 0) {
      releases.shift()?.();
    }

    await expect(movement).resolves.toEqual({
      affectedBoardIds: ['destination'],
      failed: [],
      succeeded: refs,
    });
    expect(maxActive).toBe(4);
    expect(mocks.apiFetchJson).toHaveBeenCalledTimes(refs.length);
  });

  it('stops scheduling video moves after abort and leaves aborted or unstarted refs unconfirmed', async () => {
    const controller = new AbortController();
    const refs = Array.from({ length: 8 }, (_, index) => videoRef(`clip-${index}.mp4`));

    mocks.apiFetchJson.mockImplementation((_url: string, init?: RequestInit) => {
      const signal = init?.signal;
      if (!signal) {
        throw new Error('Expected the caller abort signal');
      }

      return new Promise((_resolve, reject) => {
        const rejectAborted = () => reject(signal.reason ?? new DOMException('Aborted', 'AbortError'));

        if (signal.aborted) {
          rejectAborted();
          return;
        }
        signal.addEventListener('abort', rejectAborted, { once: true });
      });
    });

    const movement = galleryItemOrganization.moveToBoard(refs, 'destination', controller.signal);

    await vi.waitFor(() => {
      expect(mocks.apiFetchJson).toHaveBeenCalledTimes(4);
    });
    controller.abort(new DOMException('Stopped', 'AbortError'));

    await expect(movement).resolves.toEqual({
      affectedBoardIds: [],
      failed: refs,
      succeeded: [],
    });
    expect(mocks.apiFetchJson).toHaveBeenCalledTimes(4);
  });

  it.each([
    { affected_boards: ['board-a'], starred_videos: 'clip.mp4' },
    { affected_boards: ['unexpected-board'], starred_videos: ['unexpected.mp4'] },
  ])('does not confirm malformed or unexpected video success payloads: $starred_videos', async (response) => {
    mocks.apiFetchJson.mockResolvedValue(response);

    await expect(galleryItemOrganization.setStarred([videoRef('clip.mp4')], true)).resolves.toEqual({
      affectedBoardIds: [],
      failed: [videoRef('clip.mp4')],
      succeeded: [],
    });
  });

  it('does not issue transport requests for an empty operation', async () => {
    await expect(galleryItemOrganization.delete([])).resolves.toEqual({
      affectedBoardIds: [],
      failed: [],
      succeeded: [],
    });
    expect(mocks.apiFetchJson).not.toHaveBeenCalled();
  });
});
