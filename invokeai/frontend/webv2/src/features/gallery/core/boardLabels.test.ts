import { describe, expect, it } from 'vitest';

import type { GalleryBoard } from './types';

import { getGalleryBoardLabel } from './boardLabels';

const createBoard = (overrides: Partial<GalleryBoard> = {}): GalleryBoard => ({
  archived: false,
  assetCount: 3,
  id: 'dogs',
  imageCount: 50,
  kind: 'board',
  name: 'dogs',
  projectId: null,
  videoCount: 4,
  ...overrides,
});

const t = (key: string) => (key === 'widgets.gallery.uncategorized' ? 'Uncategorized' : key);

describe('getGalleryBoardLabel', () => {
  it('uses the stored name for real boards', () => {
    expect(getGalleryBoardLabel(createBoard(), t)).toBe('dogs');
  });

  it('translates the synthesized uncategorized board instead of trusting its name', () => {
    expect(getGalleryBoardLabel(createBoard({ kind: 'uncategorized', name: '' }), t)).toBe('Uncategorized');
  });
});
