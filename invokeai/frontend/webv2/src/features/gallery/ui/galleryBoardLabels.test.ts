import type { GalleryBoard } from '@features/gallery/core/types';

import { describe, expect, it } from 'vitest';

import { getGalleryBoardOwnerLabel, getGalleryCountForView } from './galleryBoardLabels';

const createBoard = (overrides: Partial<GalleryBoard> = {}): GalleryBoard => ({
  archived: false,
  assetCount: 3,
  assetVideoCount: 0,
  id: 'dogs',
  imageCount: 50,
  kind: 'board',
  name: 'dogs',
  isInbox: false,
  projectId: null,
  videoCount: 4,
  ...overrides,
});

describe('getGalleryCountForView', () => {
  it('counts images and videos together for the media view', () => {
    expect(getGalleryCountForView(createBoard(), 'images')).toBe(54);
  });

  it('counts assets alone for the assets view', () => {
    expect(getGalleryCountForView(createBoard(), 'assets')).toBe(3);
  });

  it('splits asset-category videos out of media and into assets', () => {
    // 4 videos total, 1 uploaded (asset): media shows 50 images + 3 media videos,
    // assets shows 3 asset images + 1 asset video.
    const board = createBoard({ assetVideoCount: 1 });

    expect(getGalleryCountForView(board, 'images')).toBe(53);
    expect(getGalleryCountForView(board, 'assets')).toBe(4);
  });
});

describe('getGalleryBoardOwnerLabel', () => {
  const t = (key: string, options?: Record<string, unknown>) =>
    options ? `${key}(${JSON.stringify(options)})` : key.replace('widgets.gallery.boardAccess.', '');
  const withAccess = (role: 'owner' | 'viewer' | 'contributor' | 'editor' | null, adminOverride = false) =>
    createBoard({
      access: { actions: ['read'], adminOverride, generalAccess: 'viewer', role },
      ownerName: 'Ana',
    });

  it('says whose board it is and what the caller may do, and nothing on their own', () => {
    expect(getGalleryBoardOwnerLabel(withAccess('owner'), t)).toBeNull();
    expect(getGalleryBoardOwnerLabel(withAccess('viewer'), t)).toBe(
      'widgets.gallery.boardAccess.ownerAndAccess({"access":"viewer","owner":"Ana"})'
    );
    expect(getGalleryBoardOwnerLabel(withAccess('editor'), t)).toContain('"access":"editor"');
    // Admin rights say more than a role they exceed.
    expect(getGalleryBoardOwnerLabel(withAccess('viewer', true), t)).toContain('"access":"admin"');
    expect(getGalleryBoardOwnerLabel(withAccess(null, true), t)).toContain('"access":"admin"');
    expect(getGalleryBoardOwnerLabel({ ...withAccess('viewer'), ownerName: null }, t)).toContain(
      '"owner":"deletedOwner"'
    );
  });
});
