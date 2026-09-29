import { registerExternalImageFile, registerImageCluster } from '@features/gallery/core/semanticImageQuery';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const mocks = vi.hoisted(() => ({
  apiFetchJson: vi.fn(),
}));

vi.mock('@platform/transport/http', () => ({
  absolutizeApiUrl: (url: string) => url,
  ApiError: class ApiError extends Error {},
  apiFetch: vi.fn(),
  apiFetchJson: mocks.apiFetchJson,
  apiFetchRaw: vi.fn(),
  HttpRequestIdentityExpiredError: class HttpRequestIdentityExpiredError extends Error {},
  sleep: vi.fn(),
}));

import {
  hydrateGalleryDateBoardItemPage,
  listPaletteSemanticImages,
  listSemanticGalleryItemNames,
  searchGallerySemantic,
} from './backend';
import { canonicalizeGalleryItemsFilter, type GalleryItemsFilter } from './queries';

const backendImage = (name: string) => ({
  created_at: '2026-08-02',
  height: 64,
  image_category: 'general',
  image_name: name,
  image_url: `/img/${name}`,
  is_intermediate: false,
  thumbnail_url: `/thumb/${name}`,
  width: 64,
});

describe('searchGallerySemantic', () => {
  beforeEach(() => {
    mocks.apiFetchJson.mockReset();
  });

  it('builds text queries and maps results', async () => {
    mocks.apiFetchJson.mockResolvedValue({ results: [{ image_name: 'a.png', score: 0.9 }] });

    const results = await searchGallerySemantic({ kind: 'text', query: 'red barn' }, { limit: 50 });

    expect(mocks.apiFetchJson).toHaveBeenCalledWith(
      '/api/v1/image_map/search?include_videos=true&limit=50&q=red+barn',
      {
        signal: undefined,
      }
    );
    expect(results).toEqual([{ ref: { kind: 'image', name: 'a.png' }, score: 0.9 }]);
  });

  it('carries each hit kind so videos hydrate through their own endpoint', async () => {
    mocks.apiFetchJson.mockResolvedValue({
      results: [
        { image_name: 'clip.mp4', kind: 'video', score: 0.9 },
        { image_name: 'a.png', kind: 'image', score: 0.5 },
      ],
    });

    const results = await searchGallerySemantic({ kind: 'text', query: 'surf' });

    // Reading a video hit as an image sends the gallery to /images/i/<name>,
    // which 404s — a ranked result that renders as a broken tile.
    expect(results).toEqual([
      { ref: { kind: 'video', name: 'clip.mp4' }, score: 0.9 },
      { ref: { kind: 'image', name: 'a.png' }, score: 0.5 },
    ]);
  });

  it('builds image-similarity queries with the default result cap', async () => {
    mocks.apiFetchJson.mockResolvedValue({ results: [] });

    await searchGallerySemantic({ imageName: 'ref.png', kind: 'image' });

    expect(mocks.apiFetchJson).toHaveBeenCalledWith(
      '/api/v1/image_map/search?image_name=ref.png&include_videos=true&limit=500',
      {
        signal: undefined,
      }
    );
  });

  it('POSTs url queries to the by-image endpoint', async () => {
    mocks.apiFetchJson.mockResolvedValue({ results: [{ image_name: 'a.png', score: 0.8 }] });

    const results = await searchGallerySemantic({ kind: 'url', url: 'https://example.com/cat.jpg' }, { limit: 25 });

    expect(mocks.apiFetchJson).toHaveBeenCalledWith(
      '/api/v1/image_map/search_by_image?image_url=https%3A%2F%2Fexample.com%2Fcat.jpg&include_videos=true&limit=25',
      { method: 'POST', signal: undefined }
    );
    expect(results).toEqual([{ ref: { kind: 'image', name: 'a.png' }, score: 0.8 }]);
  });

  it('POSTs registered dropped files as multipart and fails clearly when the blob is gone', async () => {
    mocks.apiFetchJson.mockResolvedValue({ results: [] });
    const fileId = registerExternalImageFile(new Blob(['not-really-a-png']), 'cat.png');

    await searchGallerySemantic({ fileId, kind: 'file' }, { limit: 10 });

    const [path, init] = mocks.apiFetchJson.mock.calls[0] as [string, RequestInit];

    expect(path).toBe('/api/v1/image_map/search_by_image?include_videos=true&limit=10');
    expect(init.method).toBe('POST');
    expect(init.body).toBeInstanceOf(FormData);
    expect((init.body as FormData).get('image')).toBeInstanceOf(Blob);

    await expect(searchGallerySemantic({ fileId: 'external-unknown', kind: 'file' })).rejects.toThrow(
      /no longer available/
    );
  });
});

describe('semantic search scope', () => {
  beforeEach(() => {
    mocks.apiFetchJson.mockReset();
    mocks.apiFetchJson.mockResolvedValue({ results: [] });
  });

  it('ranks within the gallery board, a date board by its day, and everything for the all-readable scope', async () => {
    const query = { kind: 'text', query: 'boats' } as const;

    await listSemanticGalleryItemNames({ boardId: 'board-1', query });
    await listSemanticGalleryItemNames({ boardId: 'none', query });
    await listSemanticGalleryItemNames({ boardId: 'by_date:2026-08-02', query });
    await listSemanticGalleryItemNames({ boardId: 'all', query });

    expect(mocks.apiFetchJson.mock.calls.map(([path]) => path)).toEqual([
      '/api/v1/image_map/search?board_id=board-1&include_videos=true&limit=500&q=boats',
      '/api/v1/image_map/search?board_id=none&include_videos=true&limit=500&q=boats',
      '/api/v1/image_map/search?created_date=2026-08-02&include_videos=true&limit=500&q=boats',
      '/api/v1/image_map/search?include_videos=true&limit=500&q=boats',
    ]);
  });

  it('scopes by-image searches the same way', async () => {
    const fileId = registerExternalImageFile(new Blob(['not-really-a-png']), 'cat.png');

    await searchGallerySemantic({ fileId, kind: 'file' }, { boardId: 'board-1', limit: 10 });
    await searchGallerySemantic({ kind: 'url', url: 'https://example.com/cat.jpg' }, { boardId: 'board-1', limit: 10 });

    expect(mocks.apiFetchJson.mock.calls.map(([path]) => path)).toEqual([
      '/api/v1/image_map/search_by_image?board_id=board-1&include_videos=true&limit=10',
      '/api/v1/image_map/search_by_image?image_url=https%3A%2F%2Fexample.com%2Fcat.jpg&board_id=board-1&include_videos=true&limit=10',
    ]);
  });
});

describe('listPaletteSemanticImages', () => {
  it('ranks images across the whole library and keeps rank order through hydration', async () => {
    mocks.apiFetchJson.mockReset();
    mocks.apiFetchJson
      .mockResolvedValueOnce({
        results: [
          { image_name: 'first.png', kind: 'image', score: 0.9 },
          { image_name: 'second.png', kind: 'image', score: 0.8 },
        ],
      })
      .mockResolvedValueOnce([backendImage('second.png'), backendImage('first.png')]);

    const images = await listPaletteSemanticImages({ limit: 20, query: 'boats' });

    // No board or date scope, and images only: the palette opens results in Preview.
    expect(mocks.apiFetchJson.mock.calls[0]?.[0]).toBe(
      '/api/v1/image_map/search?include_videos=false&limit=20&q=boats'
    );
    expect(images.map((image) => image.imageName)).toEqual(['first.png', 'second.png']);
  });
});

describe('semantic page hydration', () => {
  beforeEach(() => {
    mocks.apiFetchJson.mockReset();
  });

  it('slices the ranked name list and hydrates DTOs in rank order, dropping unknown names', async () => {
    mocks.apiFetchJson
      .mockResolvedValueOnce({
        results: [
          { image_name: 'first.png', score: 0.9 },
          { image_name: 'second.png', score: 0.8 },
          { image_name: 'deleted.png', score: 0.7 },
        ],
      })
      // by-names returns DTOs in arbitrary order; rank order must win, and a
      // name the hydration endpoint no longer knows is skipped.
      .mockResolvedValueOnce([backendImage('second.png'), backendImage('first.png')]);

    const names = await listSemanticGalleryItemNames({ query: { kind: 'text', query: 'boats' } });
    const page = await hydrateGalleryDateBoardItemPage({ ...names, limit: 3, offset: 0 });

    expect(page.total).toBe(3);
    expect(page.items.map((item) => item.name)).toEqual(['first.png', 'second.png']);
    expect(page.items.every((item) => item.kind === 'image')).toBe(true);
  });
});

describe('listSemanticGalleryItemNames', () => {
  it('maps the ranked results to image refs in rank order', async () => {
    mocks.apiFetchJson.mockReset();
    mocks.apiFetchJson.mockResolvedValue({
      results: [
        { image_name: 'first.png', score: 0.9 },
        { image_name: 'second.png', score: 0.8 },
      ],
    });

    await expect(listSemanticGalleryItemNames({ query: { imageName: 'x.png', kind: 'image' } })).resolves.toEqual({
      items: [
        { kind: 'image', name: 'first.png' },
        { kind: 'image', name: 'second.png' },
      ],
      total: 2,
    });
  });

  it('answers cluster queries from the registry without a server round trip', async () => {
    mocks.apiFetchJson.mockReset();
    const clusterId = registerImageCluster(['image:near.png', 'video:clip.mp4'], 'beaches');

    await expect(listSemanticGalleryItemNames({ query: { clusterId, kind: 'cluster' } })).resolves.toEqual({
      items: [
        { kind: 'image', name: 'near.png' },
        { kind: 'video', name: 'clip.mp4' },
      ],
      total: 2,
    });
    // Evicted cluster keys yield empty results; parsing clears the stale reference before display.
    registerImageCluster(['image:other.png'], 'newer');
    await expect(listSemanticGalleryItemNames({ query: { clusterId, kind: 'cluster' } })).resolves.toEqual({
      items: [],
      total: 0,
    });
    expect(mocks.apiFetchJson).not.toHaveBeenCalled();
  });
});

describe('semantic filter canonicalization', () => {
  const baseFilter: GalleryItemsFilter = {
    boardId: 'none',
    galleryView: 'images',
    searchTerm: '',
  };

  it('omits the semantic field entirely when no reference is set', () => {
    expect(canonicalizeGalleryItemsFilter(baseFilter)).not.toHaveProperty('semantic');
    expect(canonicalizeGalleryItemsFilter({ ...baseFilter, semanticQuery: null })).not.toHaveProperty('semantic');
  });

  it('discriminates cache identity by kind and keys file references by registry id', () => {
    expect(
      canonicalizeGalleryItemsFilter({ ...baseFilter, semanticQuery: { imageName: 'a.png', kind: 'image' } }).semantic
    ).toEqual({ imageName: 'a.png', kind: 'image' });

    // Same value, different kind: must not collide in the cache.
    expect(
      canonicalizeGalleryItemsFilter({ ...baseFilter, semanticQuery: { kind: 'url', url: 'a.png' } }).semantic
    ).toEqual({ kind: 'url', url: 'a.png' });

    // The label is presentation, not identity: relabeled drops of the same
    // registered file canonicalize identically.
    const withLabelA = canonicalizeGalleryItemsFilter({
      ...baseFilter,
      semanticQuery: { fileId: 'external-7', kind: 'file', label: 'a.png' },
    });
    const withLabelB = canonicalizeGalleryItemsFilter({
      ...baseFilter,
      semanticQuery: { fileId: 'external-7', kind: 'file', label: 'b.png' },
    });

    expect(withLabelA.semantic).toEqual({ fileId: 'external-7', kind: 'file' });
    expect(withLabelA).toEqual(withLabelB);
  });
});
