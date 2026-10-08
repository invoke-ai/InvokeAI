import type { GalleryImageItem, GalleryItem, GalleryVideoItem } from '@features/gallery/core/items';
import type { GalleryBoard, GeneratedImageContract } from '@features/gallery/core/types';

import { describe, expect, it } from 'vitest';

import {
  getBoardCounts,
  getGalleryAutoAddBoardId,
  getGalleryDestinationBoardId,
  getGallerySelectedBoardId,
  getGallerySelectedImageQuery,
  getGallerySemanticImageQuery,
  getGalleryListing,
  getGalleryReadStatus,
  getGalleryStateView,
  type GalleryQueryFacts,
} from './galleryStateView';

const boards: GalleryBoard[] = [
  {
    archived: false,
    assetCount: 0,
    assetVideoCount: 0,
    id: 'none',
    imageCount: 1,
    kind: 'uncategorized',
    name: '',
    projectId: null,
    videoCount: 0,
  },
  {
    archived: false,
    assetCount: 0,
    assetVideoCount: 0,
    id: 'board-1',
    imageCount: 2,
    kind: 'board',
    name: 'Board 1',
    projectId: null,
    videoCount: 0,
  },
];

const createImage = (imageName: string): GeneratedImageContract => ({
  height: 768,
  imageName,
  imageUrl: `/api/v1/images/i/${imageName}/full`,
  queuedAt: '2026-06-09T00:00:00.000Z',
  sourceQueueItemId: 'queue-item-1',
  thumbnailUrl: `/api/v1/images/i/${imageName}/thumbnail`,
  width: 512,
});

const createImageItem = (name: string): GalleryImageItem => ({
  boardId: 'none',
  category: 'general',
  createdAt: '2026-06-09T00:00:00.000Z',
  fullUrl: `/api/v1/images/i/${name}/full`,
  height: 768,
  isIntermediate: false,
  kind: 'image',
  name,
  sourceQueueItemId: 'queue-item-1',
  starred: false,
  thumbnailUrl: `/api/v1/images/i/${name}/thumbnail`,
  width: 512,
});

const createVideoItem = (name: string): GalleryVideoItem => ({
  ...createImageItem(name),
  durationSeconds: 3,
  fullUrl: `/api/v1/videos/i/${name}/full`,
  kind: 'video',
  thumbnailUrl: `/api/v1/videos/i/${name}/thumbnail`,
});

describe('gallery state view', () => {
  it('preserves a selected backend board id while boards are still loading', () => {
    const values = { selectedBoardId: 'board-1' };

    expect(getGallerySelectedBoardId(values, [])).toBe('board-1');
    expect(getGalleryStateView(values, [], null).selectedBoardId).toBe('board-1');
  });

  it('falls back to uncategorized after loaded boards do not contain the selected board', () => {
    const values = { selectedBoardId: 'missing-board' };

    expect(getGallerySelectedBoardId(values, boards)).toBe('none');
    expect(getGalleryStateView(values, boards, []).selectedBoardId).toBe('none');
  });

  /**
   * Missing imported destinations fall back to the project's board instead of scattering output into
   * Uncategorized.
   */
  it('falls back to the project board before uncategorized', () => {
    const projectBoards = [...boards, { ...boards[1]!, id: 'project-board', name: 'My Project', projectId: 'p1' }];
    const values = { projectBoardId: 'project-board', selectedBoardId: 'missing-board' };

    expect(getGallerySelectedBoardId(values, projectBoards)).toBe('project-board');
    expect(getGalleryStateView(values, projectBoards, []).selectedBoardId).toBe('project-board');
  });

  it('keeps a still-resolvable selection rather than reverting to the project board', () => {
    const projectBoards = [...boards, { ...boards[1]!, id: 'project-board', name: 'My Project', projectId: 'p1' }];
    const values = { projectBoardId: 'project-board', selectedBoardId: 'board-1' };

    expect(getGallerySelectedBoardId(values, projectBoards)).toBe('board-1');
  });

  it('falls back to uncategorized when the project board is gone too', () => {
    const values = { projectBoardId: 'deleted-board', selectedBoardId: 'missing-board' };

    expect(getGallerySelectedBoardId(values, boards)).toBe('none');
  });

  /** An absent saved destination is not an explicit choice of Uncategorized; use the project's board. */
  it('uses the project board when nothing was ever selected', () => {
    const projectBoards = [...boards, { ...boards[1]!, id: 'project-board', name: 'My Project', projectId: 'p1' }];

    expect(getGallerySelectedBoardId({ projectBoardId: 'project-board' }, projectBoards)).toBe('project-board');
    expect(getGallerySelectedBoardId({}, projectBoards)).toBe('none');
  });

  it('never substitutes unfiltered recents for a listing the current scope does not have', () => {
    // Recents from another board, an anchored window or a failed page are not this scope's items.
    const values = {
      galleryPage: 3,
      recentImages: [createImage('elsewhere.png')],
      selectedBoardId: 'board-1',
      selectedImageName: 'image:elsewhere.png',
    };
    const gallery = getGalleryStateView(values, boards, null);

    expect(gallery.items).toEqual([]);
    expect(gallery.selectedItemKey).toBeNull();
  });

  it('exposes image, video, and asset counts for board labels', () => {
    expect(getBoardCounts({ ...boards[1], videoCount: 3 })).toEqual({
      assetCount: 0,
      assetVideoCount: 0,
      imageCount: 2,
      videoCount: 3,
    });
  });

  it('parses persisted gallery settings with safe defaults', () => {
    const gallery = getGalleryStateView({ boardOrderBy: 'board_name', starredSectionCollapsed: 'yes' }, boards, []);

    expect(gallery.settings).toEqual({
      autoAddBoardId: 'follow',
      boardOrderBy: 'board_name',
      boardOrderDir: 'DESC',
      boardPanelCollapsed: false,
      boardPanelHeightPx: 280,
      boardPanelWidthPx: 240,
      collapsedBoardSections: [],
      imageDensityPercent: 50,
      imageOrderDir: 'DESC',
      paginationMode: 'infinite',
      showArchivedBoards: false,
      showDateBoards: false,
      showImageDimensions: false,
      showOtherProjectBoards: false,
      showPendingItems: true,
      progressSectionCollapsed: false,
      starredSectionCollapsed: false,
      thumbnailFit: 'square',
    });
    expect(getGalleryStateView({ starredSectionCollapsed: true }, boards, []).settings.starredSectionCollapsed).toBe(
      true
    );
  });

  it('exposes comparison state only while an image selection differs from the compare image', () => {
    const selected = createImageItem('selected.png');
    const values = {
      compareImage: createImageItem('compare.png'),
      selectedBoardId: 'none',
      selectedImageName: 'image:selected.png',
    };

    expect(getGalleryStateView(values, boards, [selected]).isComparisonActive).toBe(true);
    expect(getGalleryStateView({ ...values, compareImage: selected }, boards, [selected]).isComparisonActive).toBe(
      false
    );
    expect(getGalleryStateView(values, boards, []).isComparisonActive).toBe(false);

    const video = createVideoItem('clip');

    expect(
      getGalleryStateView({ ...values, selectedImageName: 'video:clip' }, boards, [video]).isComparisonActive
    ).toBe(false);
  });

  it('exposes the selection page only when its stamp names the listing the grid shows', () => {
    const stamped = {
      galleryPage: 0,
      paginationMode: 'paginated',
      selectedBoardId: 'none',
      selectedImageQuery: {
        boardId: 'none',
        galleryView: 'images',
        imageOrderDir: 'DESC',
        page: 2,
        paginationMode: 'paginated',
        searchTerm: '',
        starredOnly: false,
      },
    };
    const pageOf = (values: Record<string, unknown>, stamp: Record<string, unknown> = {}) =>
      getGalleryStateView(
        { ...stamped, ...values, selectedImageQuery: { ...stamped.selectedImageQuery, ...stamp } },
        boards,
        []
      ).revealTargetPage;

    expect(pageOf({})).toBe(2);
    expect(getGalleryStateView(stamped, boards, []).page).toBe(0);
    expect(pageOf({ selectedBoardId: 'board-1' })).toBeNull();
    expect(pageOf({ paginationMode: 'infinite' })).toBeNull();
    expect(pageOf({}, { galleryView: 'assets' })).toBeNull();
    expect(pageOf({}, { paginationMode: 'infinite' })).toBeNull();
    expect(pageOf({ imageOrderDir: 'ASC' })).toBeNull();
    expect(pageOf({ searchTerm: 'cats' })).toBeNull();
    expect(pageOf({ starredOnly: true })).toBeNull();
    expect(pageOf({ starredOnly: true }, { starredOnly: true })).toBe(2);
    // A starred selection sits in the strip, so no page of the unstarred grid holds it.
    expect(pageOf({ selectedImage: { ...createImageItem('starred.png'), starred: true } })).toBeNull();
    expect(
      pageOf(
        { selectedImage: { ...createImageItem('starred.png'), starred: true }, starredOnly: true },
        { starredOnly: true }
      )
    ).toBe(2);
    expect(pageOf({ semanticImageQuery: { imageName: 'ref.png', kind: 'image' } })).toBeNull();
  });

  it('reads the starred-only filter as a strict boolean and stamps it on the selection query', () => {
    expect(getGalleryStateView({ starredOnly: true }, boards, []).starredOnly).toBe(true);
    expect(getGalleryStateView({ starredOnly: 'true' }, boards, []).starredOnly).toBe(false);
    expect(getGalleryStateView({}, boards, []).starredOnly).toBe(false);

    // The stamp wins over the live value: navigation walks the listing the
    // selection was made in, not the one the grid has since switched to.
    expect(
      getGallerySelectedImageQuery({ selectedImageQuery: { starredOnly: true }, starredOnly: false })
    ).toMatchObject({ starredOnly: true });
    expect(getGallerySelectedImageQuery({ starredOnly: true })).toMatchObject({ starredOnly: true });
  });

  it('qualifies legacy names and preserves ordered mixed-media selection keys', () => {
    const gallery = getGalleryStateView(
      { selectedImageNames: ['a.png', 'video:shared', 'image:shared', 7] },
      boards,
      []
    ) as ReturnType<typeof getGalleryStateView> & { selectedItemKeys?: string[] };

    expect(gallery.selectedItemKeys).toEqual(['image:a.png', 'video:shared', 'image:shared']);
    expect(
      (
        getGalleryStateView({ selectedImageName: 'a.png' }, boards, []) as ReturnType<typeof getGalleryStateView> & {
          selectedItemKeys?: string[];
        }
      ).selectedItemKeys
    ).toEqual(['image:a.png']);
  });

  it('restores the selection set from a visible primary item after tab switches clear it', () => {
    const image = createImageItem('selected.png');
    const gallery = getGalleryStateView({ selectedImageName: 'image:selected.png', selectedImageNames: [] }, boards, [
      image,
    ]) as ReturnType<typeof getGalleryStateView> & {
      selectedItemKey?: string | null;
      selectedItemKeys?: string[];
    };

    expect(gallery.selectedItemKey).toBe('image:selected.png');
    expect(gallery.selectedItemKeys).toEqual(['image:selected.png']);
  });

  it('treats a selection in the starred strip as visible, though the listing does not hold it', () => {
    const starred = { ...createImageItem('starred.png'), starred: true };
    const values = { selectedImageName: 'image:starred.png' };

    expect(getGalleryStateView(values, boards, [createImageItem('regular.png')]).selectedItemKey).toBeNull();
    expect(getGalleryStateView(values, boards, [createImageItem('regular.png')]).primarySelectedItemKey).toBe(
      'image:starred.png'
    );
    expect(getGalleryStateView(values, boards, [createImageItem('regular.png')], [starred]).selectedItemKey).toBe(
      'image:starred.png'
    );
  });

  it('projects same-name images and videos independently by qualified key', () => {
    const image = createImageItem('shared');
    const video = createVideoItem('shared');
    const gallery = getGalleryStateView(
      {
        compareImage: image,
        selectedImage: video,
        selectedImageName: 'video:shared',
        selectedImageNames: ['image:shared', 'video:shared'],
      },
      boards,
      [image, video]
    ) as ReturnType<typeof getGalleryStateView> & {
      compareImageKey?: string | null;
      items?: GalleryItem[];
      selectedItemKey?: string | null;
      selectedItemKeys?: string[];
    };

    expect(gallery.items?.map((item) => `${item.kind}:${item.name}`)).toEqual(['image:shared', 'video:shared']);
    expect(gallery.selectedItemKey).toBe('video:shared');
    expect(gallery.selectedItemKeys).toEqual(['image:shared', 'video:shared']);
    expect(gallery.compareImageKey).toBe('image:shared');
  });

  it('parses a persisted semantic reference for the gallery view', () => {
    const values = { semanticImageQuery: { kind: 'url', url: 'https://x.test/i.png' } };
    const first = getGallerySemanticImageQuery(values);
    const second = getGallerySemanticImageQuery({ ...values });

    expect(first).toEqual({ kind: 'url', url: 'https://x.test/i.png' });
    expect(second).toEqual(first);
    expect(getGallerySemanticImageQuery({ semanticImageQuery: 'other.png' })).toEqual({
      imageName: 'other.png',
      kind: 'image',
    });
  });
  it('threads persisted semantic references into the board view', () => {
    const ranked = getGalleryStateView({ selectedBoardId: 'board-1', semanticImageQuery: 'ref.png' }, boards, []);
    expect(ranked.semanticImageQuery).toEqual({ imageName: 'ref.png', kind: 'image' });
    expect(getGalleryStateView({ selectedBoardId: 'board-1' }, boards, []).semanticImageQuery).toBeNull();
  });
});

describe('gallery read status', () => {
  const pending: GalleryQueryFacts = { errorUpdateCount: 0, hasData: false, isError: false };
  const failed: GalleryQueryFacts = { errorUpdateCount: 1, hasData: false, isError: true };
  const loaded: GalleryQueryFacts = { errorUpdateCount: 0, hasData: true, isError: false };

  it('tells a first load, an empty success and a failure apart', () => {
    expect(getGalleryReadStatus(pending, 0)).toBe('loading');
    expect(getGalleryReadStatus(loaded, 0)).toBe('empty');
    expect(getGalleryReadStatus(loaded, 3)).toBe('ready');
    expect(getGalleryReadStatus(failed, 0)).toBe('error');
  });

  it('keeps a failed scope failed while its retry is in flight', () => {
    // A refetch resets a dataless query to pending with no error; the failure count survives that reset.
    expect(getGalleryReadStatus({ errorUpdateCount: 1, hasData: false, isError: false }, 0)).toBe('error');
  });

  it("distinguishes a failed refresh from a failed next page of this scope's data", () => {
    const refreshFailed = { ...loaded, errorUpdateCount: 1, isError: true };

    expect(getGalleryReadStatus(refreshFailed, 3)).toBe('stale-error');
    expect(getGalleryReadStatus({ ...refreshFailed, isFetchNextPageError: true }, 3)).toBe('more-error');
    // A once-failed scope that has since loaded is healthy again.
    expect(getGalleryReadStatus({ ...loaded, errorUpdateCount: 2 }, 3)).toBe('ready');
  });
});

describe('getGalleryListing', () => {
  const recent = createImageItem('just-generated.png');
  const backend = createImageItem('backend.png');

  it('lets scope-filtered recents ride along while the scope loads or has data', () => {
    expect(getGalleryListing({ errorUpdateCount: 0, hasData: false, isError: false }, [recent])).toEqual({
      items: [recent],
      status: 'loading',
    });
    expect(getGalleryListing({ errorUpdateCount: 0, hasData: true, isError: false }, [recent, backend])).toEqual({
      items: [recent, backend],
      status: 'ready',
    });
  });

  it('reports nothing to show before the first result when no recents apply', () => {
    expect(getGalleryListing({ errorUpdateCount: 0, hasData: false, isError: false }, [])).toEqual({
      items: null,
      status: 'loading',
    });
  });

  it('never lets recents stand in for a scope that failed without data, even mid-retry', () => {
    expect(getGalleryListing({ errorUpdateCount: 1, hasData: false, isError: true }, [recent])).toEqual({
      items: null,
      status: 'error',
    });
    expect(getGalleryListing({ errorUpdateCount: 1, hasData: false, isError: false }, [recent])).toEqual({
      items: null,
      status: 'error',
    });
  });

  it("keeps this scope's earlier items through a failed refresh or next page", () => {
    const facts = { errorUpdateCount: 1, hasData: true, isError: true };

    expect(getGalleryListing(facts, [backend])).toEqual({ items: [backend], status: 'stale-error' });
    expect(getGalleryListing({ ...facts, isFetchNextPageError: true }, [backend])).toEqual({
      items: [backend],
      status: 'more-error',
    });
  });
});

describe('getGalleryDestinationBoardId', () => {
  it('sends results to the picked board, keeping an explicit Uncategorized choice', () => {
    expect(getGalleryDestinationBoardId({ projectBoardId: 'project', selectedBoardId: 'picked' })).toBe('picked');
    expect(getGalleryDestinationBoardId({ projectBoardId: 'project', selectedBoardId: 'none' })).toBe('none');
  });

  it('falls back to the project board when nothing is picked or a date bucket is', () => {
    expect(getGalleryDestinationBoardId({ projectBoardId: 'project' })).toBe('project');
    expect(getGalleryDestinationBoardId({ projectBoardId: 'project', selectedBoardId: 'by_date:2026-07-15' })).toBe(
      'project'
    );
    expect(getGalleryDestinationBoardId({})).toBeNull();
  });
});

describe('getGalleryAutoAddBoardId', () => {
  it('follows the gallery destination until a board is fixed', () => {
    expect(getGalleryAutoAddBoardId({ projectBoardId: 'project', selectedBoardId: 'picked' })).toBe('picked');
    expect(getGalleryAutoAddBoardId({ autoAddBoardId: 'follow', projectBoardId: 'project' })).toBe('project');
  });

  it('uses a fixed board, including Uncategorized, whatever the selection', () => {
    expect(getGalleryAutoAddBoardId({ autoAddBoardId: 'fixed', selectedBoardId: 'picked' })).toBe('fixed');
    expect(getGalleryAutoAddBoardId({ autoAddBoardId: 'none', selectedBoardId: 'picked' })).toBe('none');
  });
});
