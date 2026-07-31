import type { GalleryItemRef } from '@features/gallery/core/items';

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

import { hydrateGalleryDateBoardItemPage, listGalleryDateBoardItemNames, listGalleryItemNames } from './backend';

const imageDto = (name: string) => ({
  board_id: 'board-1',
  created_at: '2026-07-30T12:00:00.000Z',
  height: 64,
  image_category: 'general',
  image_name: name,
  image_url: `/images/${name}`,
  is_intermediate: false,
  starred: false,
  thumbnail_url: `/thumbnails/${name}`,
  width: 64,
});

const videoDto = (name: string) => ({
  board_id: 'board-1',
  created_at: '2026-07-30T12:00:00.000Z',
  duration: 2,
  height: 64,
  is_intermediate: false,
  starred: false,
  thumbnail_url: `/thumbnails/${name}`,
  video_category: 'general',
  video_name: name,
  video_url: `/videos/${name}`,
  width: 64,
});

describe('Gallery item names and date hydration', () => {
  beforeEach(() => {
    mocks.apiFetchJson.mockReset();
  });

  it('uses item_names for date boards and maps ordered refs and counts', async () => {
    mocks.apiFetchJson.mockResolvedValueOnce({
      items: [
        { kind: 'video', name: 'clip.mp4' },
        { kind: 'image', name: 'still.png' },
      ],
      starred_count: 1,
      total_count: 2,
    });

    await expect(
      listGalleryDateBoardItemNames({
        boardId: 'by_date:2026-07-30',
        galleryView: 'images',
        orderDir: 'DESC',
        searchTerm: '',
        starredFirst: true,
      })
    ).resolves.toEqual({
      items: [
        { kind: 'video', name: 'clip.mp4' },
        { kind: 'image', name: 'still.png' },
      ],
      starredCount: 1,
      total: 2,
    });

    expect(mocks.apiFetchJson.mock.calls[0]?.[0]).toContain('/by_date/2026-07-30/item_names?');
    expect(mocks.apiFetchJson.mock.calls[0]?.[0]).not.toContain('image_names');
  });

  it('forwards created ranges to normal item-name requests', async () => {
    mocks.apiFetchJson.mockResolvedValueOnce({ items: [], starred_count: 0, total_count: 0 });

    await listGalleryItemNames({
      boardId: 'board-1',
      createdFrom: '2026-07-01',
      createdTo: '2026-07-30',
      galleryView: 'images',
      orderDir: 'DESC',
      searchTerm: ' portrait ',
      starredFirst: false,
    });

    const url = mocks.apiFetchJson.mock.calls[0]?.[0] as string;
    const params = new URLSearchParams(url.split('?')[1]);
    expect(url.split('?')[0]).toBe('/api/v1/gallery/items/names');
    expect(params.get('created_from')).toBe('2026-07-01');
    expect(params.get('created_to')).toBe('2026-07-30');
    expect(params.get('search_term')).toBe('portrait');
  });

  it('preserves mixed ref order, batches images once, and limits video hydration concurrency to six', async () => {
    const refs: GalleryItemRef[] = [
      { kind: 'video', name: 'video-0' },
      { kind: 'image', name: 'same-name' },
      { kind: 'video', name: 'same-name' },
      ...Array.from({ length: 11 }, (_, index) => ({ kind: 'video' as const, name: `video-${index + 1}` })),
      { kind: 'image', name: 'image-2' },
    ];
    let activeVideos = 0;
    let maxActiveVideos = 0;
    const releases: (() => void)[] = [];

    mocks.apiFetchJson.mockImplementation((url: string, init?: RequestInit) => {
      if (url === '/api/v1/images/images_by_names') {
        expect(JSON.parse(String(init?.body))).toEqual({ image_names: ['same-name', 'image-2'] });
        return Promise.resolve([imageDto('image-2'), imageDto('same-name')]);
      }

      const name = decodeURIComponent(url.split('/').at(-1) ?? '');
      activeVideos += 1;
      maxActiveVideos = Math.max(maxActiveVideos, activeVideos);
      return new Promise((resolve) => {
        releases.push(() => {
          activeVideos -= 1;
          resolve(videoDto(name));
        });
      });
    });

    const hydration = hydrateGalleryDateBoardItemPage({
      items: refs,
      limit: refs.length,
      offset: 0,
      total: refs.length,
    });

    await vi.waitFor(() => {
      expect(releases).toHaveLength(6);
    });
    while (releases.length > 0) {
      releases.shift()?.();
      await Promise.resolve();
    }
    await expect(hydration).resolves.toMatchObject({ total: refs.length });
    const page = await hydration;

    expect(maxActiveVideos).toBe(6);
    expect(page.items.map(({ kind, name }) => `${kind}:${name}`)).toEqual(
      refs.map(({ kind, name }) => `${kind}:${name}`)
    );
    expect(mocks.apiFetchJson.mock.calls.filter(([url]) => url === '/api/v1/images/images_by_names')).toHaveLength(1);
  });

  it('omits only 404 video refs while preserving surrounding order', async () => {
    mocks.apiFetchJson.mockImplementation((url: string) => {
      if (url === '/api/v1/images/images_by_names') {
        return Promise.resolve([imageDto('still')]);
      }
      if (url.endsWith('/missing')) {
        return Promise.reject(new MockApiError('missing', 404));
      }
      return Promise.resolve(videoDto(url.split('/').at(-1) ?? ''));
    });

    const page = await hydrateGalleryDateBoardItemPage({
      items: [
        { kind: 'video', name: 'first' },
        { kind: 'video', name: 'missing' },
        { kind: 'image', name: 'still' },
        { kind: 'video', name: 'last' },
      ],
      limit: 4,
      offset: 0,
      total: 4,
    });

    expect(page.items.map(({ kind, name }) => `${kind}:${name}`)).toEqual(['video:first', 'image:still', 'video:last']);
    expect(page.total).toBe(4);
  });

  it.each([
    new MockApiError('unauthorized', 401),
    new TypeError('network failed'),
    new DOMException('aborted', 'AbortError'),
  ])('propagates non-404 hydration errors: %s', async (error) => {
    mocks.apiFetchJson.mockRejectedValue(error);

    await expect(
      hydrateGalleryDateBoardItemPage({
        items: [{ kind: 'video', name: 'clip' }],
        limit: 1,
        offset: 0,
        total: 1,
      })
    ).rejects.toBe(error);
  });
});
