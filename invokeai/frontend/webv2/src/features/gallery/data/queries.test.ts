import { QueryClient } from '@tanstack/react-query';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const backend = vi.hoisted(() => ({
  listGalleryBoards: vi.fn(),
  listGalleryDateBoards: vi.fn(),
  listGalleryImages: vi.fn(),
}));

vi.mock('./backend', () => backend);

import { galleryBoardsOptions, galleryImagesOptions } from './queries';

describe('Gallery query read model', () => {
  beforeEach(() => {
    backend.listGalleryBoards.mockReset();
    backend.listGalleryDateBoards.mockReset();
    backend.listGalleryImages.mockReset();
  });

  it('coalesces board readers through one query key and includes requested virtual boards', async () => {
    const queryClient = new QueryClient();
    const board = { id: 'none', kind: 'uncategorized', name: 'Uncategorized' };
    const dateBoard = { id: 'by_date:2026-07-18', kind: 'date', name: 'July 18' };
    const options = galleryBoardsOptions({ includeDateBoards: true, orderDir: 'DESC' });

    backend.listGalleryBoards.mockResolvedValue([board]);
    backend.listGalleryDateBoards.mockResolvedValue([dateBoard]);

    await expect(Promise.all([queryClient.fetchQuery(options), queryClient.fetchQuery(options)])).resolves.toEqual([
      [board, dateBoard],
      [board, dateBoard],
    ]);
    expect(backend.listGalleryBoards).toHaveBeenCalledOnce();
    expect(backend.listGalleryDateBoards).toHaveBeenCalledOnce();
  });

  it('keeps pagination, search, and refresh identity in the image query key', () => {
    const base = {
      boardId: 'board-1',
      galleryView: 'images' as const,
      limit: 60,
      offset: 0,
      orderDir: 'DESC' as const,
      revision: 'images:1',
      searchTerm: 'portrait',
      starredFirst: true,
    };

    expect(galleryImagesOptions(base).queryKey).not.toEqual(galleryImagesOptions({ ...base, offset: 60 }).queryKey);
    expect(galleryImagesOptions(base).queryKey).not.toEqual(
      galleryImagesOptions({ ...base, revision: 'images:2' }).queryKey
    );
  });
});
