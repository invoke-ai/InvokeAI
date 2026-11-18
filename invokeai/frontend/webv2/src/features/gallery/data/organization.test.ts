import type { GalleryItemRef } from '@features/gallery/core/items';

import { galleryItemOrganization, isGalleryBoardAttachable } from '@features/gallery/index';
import { accountLifecycle } from '@platform/state/accountLifecycle';
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

const MockHttpRequestIdentityExpiredError = vi.hoisted(
  () =>
    class HttpRequestIdentityExpiredError extends Error {
      constructor() {
        super('The identity lifetime that started this HTTP request is no longer active.');
        this.name = 'HttpRequestIdentityExpiredError';
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
  HttpRequestIdentityExpiredError: MockHttpRequestIdentityExpiredError,
  sleep: vi.fn(),
}));

const imageRef = (name: string): GalleryItemRef => ({ kind: 'image', name });
const videoRef = (name: string): GalleryItemRef => ({ kind: 'video', name });

describe('galleryItemOrganization transport and confirmed outcomes', () => {
  beforeEach(() => {
    accountLifecycle.activate('organization-test-user');
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
      .mockResolvedValueOnce({ affected_boards: ['board-a'], failed_videos: [], starred_videos: ['clip.mp4'] })
      .mockResolvedValueOnce({ affected_boards: ['board-a'], failed_videos: [], unstarred_videos: ['clip.mp4'] })
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

  it.each([
    {
      label: 'star with missing failed_videos',
      mutate: () => galleryItemOrganization.setStarred([videoRef('clip.mp4')], true),
      response: { affected_boards: ['board-a'], starred_videos: ['clip.mp4'] },
    },
    {
      label: 'star with wrong-type failed_videos',
      mutate: () => galleryItemOrganization.setStarred([videoRef('clip.mp4')], true),
      response: { affected_boards: ['board-a'], failed_videos: 'locked.mp4', starred_videos: ['clip.mp4'] },
    },
    {
      label: 'unstar with missing failed_videos',
      mutate: () => galleryItemOrganization.setStarred([videoRef('clip.mp4')], false),
      response: { affected_boards: ['board-a'], unstarred_videos: ['clip.mp4'] },
    },
    {
      label: 'unstar with wrong-type failed_videos',
      mutate: () => galleryItemOrganization.setStarred([videoRef('clip.mp4')], false),
      response: { affected_boards: ['board-a'], failed_videos: 'locked.mp4', unstarred_videos: ['clip.mp4'] },
    },
    {
      label: 'delete with missing failed_videos',
      mutate: () => galleryItemOrganization.delete([videoRef('clip.mp4')]),
      response: { affected_boards: ['board-a'], deleted_videos: ['clip.mp4'] },
    },
    {
      label: 'delete with wrong-type failed_videos',
      mutate: () => galleryItemOrganization.delete([videoRef('clip.mp4')]),
      response: { affected_boards: ['board-a'], deleted_videos: ['clip.mp4'], failed_videos: 'locked.mp4' },
    },
  ])('does not confirm a video bulk success for $label', async ({ mutate, response }) => {
    mocks.apiFetchJson.mockResolvedValue(response);

    await expect(mutate()).resolves.toEqual({
      affectedBoardIds: [],
      failed: [videoRef('clip.mp4')],
      succeeded: [],
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
        failed_videos: [],
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
      signal: expect.any(AbortSignal),
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
      signal: expect.any(AbortSignal),
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

  it.each([
    {
      error: () => new MockApiError('Unauthorized', 401),
      label: 'HTTP 401',
    },
    {
      error: () => new MockHttpRequestIdentityExpiredError(),
      label: 'expired HTTP identity',
    },
  ])(
    'stops scheduling after fatal $label and discards all video outcomes while retaining the image partition',
    async ({ error }) => {
      const refs = [imageRef('still.png'), ...Array.from({ length: 8 }, (_, index) => videoRef(`clip-${index}.mp4`))];
      const initialVideoReleases: (() => void)[] = [];
      let rejectFatal: ((error: unknown) => void) | undefined;
      let videoRequests = 0;

      mocks.apiFetchJson.mockImplementation((url: string, init?: RequestInit) => {
        if (url === '/api/v1/board_images/batch') {
          return Promise.resolve({
            added_images: ['still.png'],
            affected_boards: ['source-image', 'destination'],
          });
        }

        const { video_name: videoName } = JSON.parse(String(init?.body)) as { video_name: string };
        const videoIndex = Number(videoName.slice('clip-'.length, -'.mp4'.length));

        videoRequests += 1;

        if (videoIndex === 0) {
          return new Promise((_resolve, reject) => {
            rejectFatal = reject;
          });
        }
        if (videoIndex < 4) {
          return new Promise((resolve) => {
            initialVideoReleases.push(() => {
              resolve({
                added_videos: [videoName],
                affected_boards: ['source-video', 'destination'],
              });
            });
          });
        }

        return Promise.resolve({
          added_videos: [videoName],
          affected_boards: ['must-not-be-reported'],
        });
      });

      const movement = galleryItemOrganization.moveToBoard(refs, 'destination');

      await vi.waitFor(() => {
        expect(videoRequests).toBe(4);
        expect(rejectFatal).toBeDefined();
        expect(initialVideoReleases).toHaveLength(3);
      });
      rejectFatal?.(error());
      await Promise.resolve();
      for (const release of initialVideoReleases) {
        release();
      }

      await expect(movement).resolves.toEqual({
        affectedBoardIds: ['destination', 'source-image'],
        failed: refs.slice(1),
        succeeded: [imageRef('still.png')],
      });
      expect(videoRequests).toBe(4);
    }
  );

  it('binds video movement to the captured account lifetime when the caller omits a signal', async () => {
    const refs = [imageRef('still.png'), ...Array.from({ length: 8 }, (_, index) => videoRef(`clip-${index}.mp4`))];
    let videoRequests = 0;

    mocks.apiFetchJson.mockImplementation((url: string, init?: RequestInit) => {
      if (url === '/api/v1/board_images/batch') {
        return Promise.resolve({
          added_images: ['still.png'],
          affected_boards: ['source-image', 'destination'],
        });
      }

      videoRequests += 1;
      const signal = init?.signal;

      if (!signal) {
        return Promise.reject(new Error('Video movement did not capture the account lifetime signal.'));
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

    const movement = galleryItemOrganization.moveToBoard(refs, 'destination');

    await vi.waitFor(() => {
      expect(videoRequests).toBe(4);
    });
    accountLifecycle.activate('replacement-user');

    await expect(movement).resolves.toEqual({
      affectedBoardIds: ['destination', 'source-image'],
      failed: refs.slice(1),
      succeeded: [imageRef('still.png')],
    });
    expect(videoRequests).toBe(4);
  });

  it.each([403, 500])('keeps an isolated HTTP %s video move failure item-local', async (status) => {
    const refs = Array.from({ length: 6 }, (_, index) => videoRef(`clip-${index}.mp4`));

    mocks.apiFetchJson.mockImplementation((_url: string, init?: RequestInit) => {
      const { video_name: videoName } = JSON.parse(String(init?.body)) as { video_name: string };

      if (videoName === 'clip-0.mp4') {
        return Promise.reject(new MockApiError('Item move failed', status));
      }

      return Promise.resolve({
        added_videos: [videoName],
        affected_boards: ['destination'],
      });
    });

    await expect(galleryItemOrganization.moveToBoard(refs, 'destination')).resolves.toEqual({
      affectedBoardIds: ['destination'],
      failed: [videoRef('clip-0.mp4')],
      succeeded: refs.slice(1),
    });
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
    { affected_boards: ['board-a'], failed_videos: [], starred_videos: 'clip.mp4' },
    { affected_boards: ['unexpected-board'], failed_videos: [], starred_videos: ['unexpected.mp4'] },
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

  it.each(['by_date:2026-08-14', 'generated', 'assets'])(
    'reports every ref as failed when moving to the virtual board %s',
    async (boardId) => {
      await expect(galleryItemOrganization.moveToBoard([videoRef('clip.mp4')], boardId)).resolves.toEqual({
        affectedBoardIds: [],
        failed: [videoRef('clip.mp4')],
        succeeded: [],
      });
      expect(mocks.apiFetchJson).not.toHaveBeenCalled();
    }
  );
});

describe('galleryItemOrganization past the server’s per-request name cap', () => {
  // As the server answers: a body naming more than 1,000 items is refused whole, before any item is touched.
  const SERVER_MAX_NAMES = 1000;
  const fields: Record<string, string> = {
    '/api/v1/board_images/batch': 'added_images',
    '/api/v1/board_images/batch/delete': 'removed_images',
    '/api/v1/images/delete': 'deleted_images',
    '/api/v1/images/star': 'starred_images',
    '/api/v1/images/unstar': 'unstarred_images',
    '/api/v1/videos/delete': 'deleted_videos',
    '/api/v1/videos/star': 'starred_videos',
    '/api/v1/videos/unstar': 'unstarred_videos',
  };
  const serve = (url: string, init: { body: string }) => {
    const body = JSON.parse(init.body) as { image_names?: string[]; video_names?: string[] };
    const names = body.image_names ?? body.video_names ?? [];

    if (names.length > SERVER_MAX_NAMES) {
      return Promise.reject(new MockApiError('List should have at most 1000 items', 422));
    }

    return Promise.resolve({ affected_boards: ['board-1'], failed_videos: [], [fields[url]!]: names });
  };
  const refs = (kind: 'image' | 'video', count: number): GalleryItemRef[] =>
    Array.from({ length: count }, (_, index) => ({ kind, name: `${kind}-${index}` }));
  const names = (items: readonly GalleryItemRef[]) => items.map((item) => item.name);
  /** The name lists sent to one endpoint, in the order sent. */
  const sentTo = (endpoint: string): string[][] =>
    mocks.apiFetchJson.mock.calls
      .filter(([url]) => url === endpoint)
      .map(([, init]) => Object.values(JSON.parse(String(init.body)) as Record<string, string[]>)[0]!);

  beforeEach(() => {
    accountLifecycle.activate('organization-test-user');
    mocks.apiFetchJson.mockReset();
    mocks.apiFetchJson.mockImplementation(serve);
  });

  it('deletes every item of a selection larger than one request may name', async () => {
    const selection = [...refs('image', 1500), ...refs('video', 1200)];

    const result = await galleryItemOrganization.delete(selection);

    expect(result).toEqual({ affectedBoardIds: ['board-1'], failed: [], succeeded: selection });
    expect(sentTo('/api/v1/images/delete')).toEqual([
      names(refs('image', 1500).slice(0, 1000)),
      names(refs('image', 1500).slice(1000)),
    ]);
    expect(sentTo('/api/v1/videos/delete')).toEqual([
      names(refs('video', 1200).slice(0, 1000)),
      names(refs('video', 1200).slice(1000)),
    ]);
  });

  it('sends 1,000 items of a kind in one request and 1,001 in two', async () => {
    await galleryItemOrganization.delete(refs('image', 1000));
    await galleryItemOrganization.delete(refs('video', 1001));

    expect(sentTo('/api/v1/images/delete').map((batch) => batch.length)).toEqual([1000]);
    expect(sentTo('/api/v1/videos/delete').map((batch) => batch.length)).toEqual([1000, 1]);
  });

  it('sends a repeated item once, whichever batch it would fall in', async () => {
    const selection = [...refs('image', 1200), ...refs('image', 1200)];

    const result = await galleryItemOrganization.delete(selection);

    expect(sentTo('/api/v1/images/delete').map((batch) => batch.length)).toEqual([1000, 200]);
    expect(new Set(sentTo('/api/v1/images/delete').flat()).size).toBe(1200);
    expect(result).toEqual({ affectedBoardIds: ['board-1'], failed: [], succeeded: refs('image', 1200) });
  });

  it('confirms what earlier batches deleted when a later one fails, for images and videos alike', async () => {
    mocks.apiFetchJson.mockImplementation((url: string, init: { body: string }) =>
      sentTo(url).length === 2 ? Promise.reject(new MockApiError('Server error', 500)) : serve(url, init)
    );
    const images = refs('image', 2500);
    const videos = refs('video', 2500);

    const result = await galleryItemOrganization.delete([...images, ...videos]);

    expect(result).toEqual({
      affectedBoardIds: ['board-1'],
      failed: [...images.slice(1000), ...videos.slice(1000)],
      succeeded: [...images.slice(0, 1000), ...videos.slice(0, 1000)],
    });
    // Nothing is sent after a failed batch.
    expect(sentTo('/api/v1/images/delete')).toHaveLength(2);
    expect(sentTo('/api/v1/videos/delete')).toHaveLength(2);
  });

  it('stops and confirms nothing once the account changes mid-run', async () => {
    mocks.apiFetchJson.mockImplementation((url: string, init: { body: string }) => {
      if (mocks.apiFetchJson.mock.calls.length === 1) {
        accountLifecycle.activate('replacement-user');
      }
      return serve(url, init);
    });
    const images = refs('image', 2500);

    const result = await galleryItemOrganization.delete(images);

    expect(result).toEqual({ affectedBoardIds: [], failed: images, succeeded: [] });
    expect(mocks.apiFetchJson).toHaveBeenCalledTimes(1);
  });

  it.each([
    ['star', () => galleryItemOrganization.setStarred(refs('image', 2500), true)],
    ['unstar', () => galleryItemOrganization.setStarred(refs('image', 2500), false)],
    ['move to a board', () => galleryItemOrganization.moveToBoard(refs('image', 2500), 'board-1')],
    ['remove from its board', () => galleryItemOrganization.moveToBoard(refs('image', 2500), 'none')],
  ])('confirms every image when asked to %s 2,500 at once', async (_operation, run) => {
    const result = await run();

    expect(result.failed).toEqual([]);
    expect(result.succeeded).toEqual(refs('image', 2500));
    expect(mocks.apiFetchJson).toHaveBeenCalledTimes(3);
  });
});

describe('isGalleryBoardAttachable', () => {
  it.each(['by_date:2026-08-14', 'generated', 'assets', 'none'])('rejects the virtual destination %s', (boardId) => {
    expect(isGalleryBoardAttachable(boardId)).toBe(false);
  });

  it.each(['board-1', 'some-uuid'])('accepts the real board %s', (boardId) => {
    expect(isGalleryBoardAttachable(boardId)).toBe(true);
  });
});
