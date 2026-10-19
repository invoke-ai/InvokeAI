import { describe, expect, it } from 'vitest';

import type { GalleryBoard } from './types';

import { asSingleAccountBoards, isOtherUsersGalleryBoard, isOwnGalleryBoard } from './boardAccess';

const board = (role: 'owner' | 'viewer' | null): GalleryBoard => ({
  access: { actions: ['read', 'add_media'], adminOverride: role !== 'owner', generalAccess: 'restricted', role },
  archived: false,
  assetCount: 0,
  assetVideoCount: 0,
  id: String(role),
  imageCount: 0,
  isInbox: false,
  kind: 'board',
  name: 'B',
  projectId: null,
  videoCount: 0,
});

describe('asSingleAccountBoards', () => {
  it('reads every board as the caller’s own, keeping what the server allows', () => {
    const [own, other, shared] = asSingleAccountBoards([board('owner'), board(null), board('viewer')]);

    expect(isOtherUsersGalleryBoard(board(null))).toBe(true);
    expect([own, other, shared].every((folded) => folded !== undefined && isOwnGalleryBoard(folded))).toBe(true);
    expect(other?.access).toMatchObject({ actions: ['read', 'add_media'], adminOverride: false });
  });
});
