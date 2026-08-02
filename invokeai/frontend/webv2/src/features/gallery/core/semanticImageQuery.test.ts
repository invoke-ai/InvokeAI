import { describe, expect, it } from 'vitest';

import {
  gallerySemanticReferenceKey,
  getExternalImageFile,
  parseGallerySemanticReference,
  registerExternalImageFile,
  semanticReferenceFromDataTransfer,
  toGallerySemanticQuery,
} from './semanticImageQuery';

describe('external image registry', () => {
  it('holds a single slot: registering a new file evicts the previous one', () => {
    const firstId = registerExternalImageFile(new Blob(['first']), 'first.png');

    expect(getExternalImageFile(firstId)).toMatchObject({ label: 'first.png' });

    const secondId = registerExternalImageFile(new Blob(['second']), 'second.png');

    expect(secondId).not.toBe(firstId);
    expect(getExternalImageFile(firstId)).toBe(null);
    expect(getExternalImageFile(secondId)).toMatchObject({ label: 'second.png' });
  });
});

describe('semanticReferenceFromDataTransfer', () => {
  const dataTransfer = (files: File[], uriList = '') => ({
    files,
    getData: (format: string) => (format === 'text/uri-list' ? uriList : ''),
  });

  it('registers dropped image files (winning over URLs) and skips non-images', () => {
    const file = new File(['x'], 'photo.png', { type: 'image/png' });
    const reference = semanticReferenceFromDataTransfer(dataTransfer([file], 'https://a.test/i.png'));

    expect(reference).toMatchObject({ kind: 'file', label: 'photo.png' });

    const nonImage = new File(['x'], 'notes.txt', { type: 'text/plain' });

    expect(semanticReferenceFromDataTransfer(dataTransfer([nonImage]))).toBe(null);
  });

  it("converts this app's own image URLs into by-name queries", () => {
    // Dragging the preview image in must not round-trip through the URL
    // downloader (the server would refuse fetching itself).
    const reference = semanticReferenceFromDataTransfer(
      dataTransfer([], 'http://localhost:5174/api/v1/images/i/ab%20c.png/full')
    );

    expect(reference).toEqual({ imageName: 'ab c.png', kind: 'image' });
  });

  it('passes external http(s) URLs through and rejects other schemes', () => {
    expect(semanticReferenceFromDataTransfer(dataTransfer([], 'https://cdn.test/cat.jpg\n# comment'))).toEqual({
      kind: 'url',
      url: 'https://cdn.test/cat.jpg',
    });
    expect(semanticReferenceFromDataTransfer(dataTransfer([], 'data:image/png;base64,xxxx'))).toBe(null);
    expect(semanticReferenceFromDataTransfer(dataTransfer([]))).toBe(null);
  });
});

describe('parseGallerySemanticReference', () => {
  it('parses legacy strings, structured references, and drops dangling file keys', () => {
    expect(parseGallerySemanticReference('a.png')).toEqual({ imageName: 'a.png', kind: 'image' });
    expect(parseGallerySemanticReference({ kind: 'text', query: 'a sunset' })).toEqual({
      kind: 'text',
      query: 'a sunset',
    });
    expect(parseGallerySemanticReference({ kind: 'text', query: '' })).toBe(null);
    expect(parseGallerySemanticReference({ kind: 'url', url: 'https://x.test/i.png' })).toEqual({
      kind: 'url',
      url: 'https://x.test/i.png',
    });

    const fileId = registerExternalImageFile(new Blob(['x']), 'dropped.png');

    expect(parseGallerySemanticReference({ fileId, kind: 'file', label: 'dropped.png' })).toEqual({
      fileId,
      kind: 'file',
      label: 'dropped.png',
    });
    // Registering evicts prior entries, so a stale key reads as no search.
    registerExternalImageFile(new Blob(['y']), 'newer.png');
    expect(parseGallerySemanticReference({ fileId, kind: 'file', label: 'dropped.png' })).toBe(null);

    expect(parseGallerySemanticReference(undefined)).toBe(null);
    expect(parseGallerySemanticReference({ kind: 'nonsense' })).toBe(null);
    expect(parseGallerySemanticReference('')).toBe(null);
  });
});

describe('semantic reference identity', () => {
  it('keys each kind distinctly and identifies file references by registry id, not label', () => {
    expect(gallerySemanticReferenceKey(null)).toBe('');
    expect(gallerySemanticReferenceKey({ imageName: 'a.png', kind: 'image' })).not.toBe(
      gallerySemanticReferenceKey({ kind: 'url', url: 'a.png' })
    );
    expect(gallerySemanticReferenceKey({ kind: 'text', query: 'a.png' })).not.toBe(
      gallerySemanticReferenceKey({ imageName: 'a.png', kind: 'image' })
    );
    expect(gallerySemanticReferenceKey({ fileId: 'external-1', kind: 'file', label: 'a' })).toBe(
      gallerySemanticReferenceKey({ fileId: 'external-1', kind: 'file', label: 'b' })
    );
  });

  it('canonicalizes a file reference to its registry id, dropping the display label', () => {
    expect(toGallerySemanticQuery({ fileId: 'external-9', kind: 'file', label: 'cat.png' })).toEqual({
      fileId: 'external-9',
      kind: 'file',
    });
    expect(toGallerySemanticQuery({ imageName: 'a.png', kind: 'image' })).toEqual({
      imageName: 'a.png',
      kind: 'image',
    });
    expect(toGallerySemanticQuery({ kind: 'url', url: 'https://x.test/i.png' })).toEqual({
      kind: 'url',
      url: 'https://x.test/i.png',
    });
    expect(toGallerySemanticQuery({ kind: 'text', query: 'a sunset' })).toEqual({
      kind: 'text',
      query: 'a sunset',
    });
  });
});
