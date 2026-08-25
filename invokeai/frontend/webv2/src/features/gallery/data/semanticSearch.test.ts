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

import { hydrateGalleryDateBoardItemPage, listSemanticGalleryItemNames, searchGallerySemantic } from './backend';
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

    expect(mocks.apiFetchJson).toHaveBeenCalledWith('/api/v1/image_map/search?limit=50&q=red+barn', {
      signal: undefined,
    });
    expect(results).toEqual([{ imageName: 'a.png', score: 0.9 }]);
  });

  it('builds image-similarity queries with the default result cap', async () => {
    mocks.apiFetchJson.mockResolvedValue({ results: [] });

    await searchGallerySemantic({ imageName: 'ref.png', kind: 'image' });

    expect(mocks.apiFetchJson).toHaveBeenCalledWith('/api/v1/image_map/search?image_name=ref.png&limit=500', {
      signal: undefined,
    });
  });

  it('POSTs url queries to the by-image endpoint', async () => {
    mocks.apiFetchJson.mockResolvedValue({ results: [{ image_name: 'a.png', score: 0.8 }] });

    const results = await searchGallerySemantic({ kind: 'url', url: 'https://example.com/cat.jpg' }, { limit: 25 });

    expect(mocks.apiFetchJson).toHaveBeenCalledWith(
      '/api/v1/image_map/search_by_image?image_url=https%3A%2F%2Fexample.com%2Fcat.jpg&limit=25',
      { method: 'POST', signal: undefined }
    );
    expect(results).toEqual([{ imageName: 'a.png', score: 0.8 }]);
  });

  it('POSTs registered dropped files as multipart and fails clearly when the blob is gone', async () => {
    mocks.apiFetchJson.mockResolvedValue({ results: [] });
    const fileId = registerExternalImageFile(new Blob(['not-really-a-png']), 'cat.png');

    await searchGallerySemantic({ fileId, kind: 'file' }, { limit: 10 });

    const [path, init] = mocks.apiFetchJson.mock.calls[0] as [string, RequestInit];

    expect(path).toBe('/api/v1/image_map/search_by_image?limit=10');
    expect(init.method).toBe('POST');
    expect(init.body).toBeInstanceOf(FormData);
    expect((init.body as FormData).get('image')).toBeInstanceOf(Blob);

    await expect(searchGallerySemantic({ fileId: 'external-unknown', kind: 'file' })).rejects.toThrow(
      /no longer available/
    );
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
      starredCount: 0,
      total: 2,
    });
  });

  it('answers cluster queries from the registry without a server round trip', async () => {
    mocks.apiFetchJson.mockReset();
    const clusterId = registerImageCluster(['near.png', 'far.png'], 'beaches');

    await expect(listSemanticGalleryItemNames({ query: { clusterId, kind: 'cluster' } })).resolves.toEqual({
      items: [
        { kind: 'image', name: 'near.png' },
        { kind: 'image', name: 'far.png' },
      ],
      starredCount: 0,
      total: 2,
    });
    // An evicted key (another cluster registered, or a reload) degrades to an
    // empty list rather than an error — the parse layer clears the reference
    // before the UI would ever show that.
    registerImageCluster(['other.png'], 'newer');
    await expect(listSemanticGalleryItemNames({ query: { clusterId, kind: 'cluster' } })).resolves.toEqual({
      items: [],
      starredCount: 0,
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
