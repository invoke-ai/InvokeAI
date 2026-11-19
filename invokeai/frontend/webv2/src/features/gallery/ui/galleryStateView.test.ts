import type { GalleryImageItem, GalleryItem, GalleryVideoItem } from '@features/gallery/core/items';
import type { GalleryBoard, GeneratedImageContract } from '@features/gallery/core/types';

import { resolvedValues } from '@platform/core/resolvedValues.testing';
import { describe, expect, it } from 'vitest';

import {
  getGalleryAutoAddBoardId,
  getGalleryDestinationBoardId,
  getGallerySelectedBoardId,
  getGallerySemanticImageQuery,
  getGalleryListing,
  getGalleryReadStatus,
  getGalleryStateView,
  type GalleryQueryFacts,
  settleGalleryValues,
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
    isInbox: false,
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
    isInbox: false,
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
    expect(getGalleryStateView(resolvedValues(values), [], null).selectedBoardId).toBe('board-1');
  });

  /** Missing imported destinations fall back to the project's board instead of scattering output into Uncategorized. */
  it('falls back to the project board before uncategorized', () => {
    const projectBoards = [
      ...boards,
      { ...boards[1]!, id: 'project-board', isInbox: true, name: 'My Project', projectId: 'p1' },
    ];
    const values = { projectBoardId: 'project-board', selectedBoardId: 'missing-board' };

    expect(getGallerySelectedBoardId(values, projectBoards)).toBe('project-board');
    expect(getGalleryStateView(resolvedValues(values), projectBoards, []).selectedBoardId).toBe('project-board');
  });

  it('never substitutes unfiltered recents for a listing the current scope does not have', () => {
    // Recents from another board, an anchored window or a failed page are not this scope's items.
    const values = {
      galleryPage: 3,
      recentImages: [createImage('elsewhere.png')],
      selectedBoardId: 'board-1',
      selectedImageName: 'image:elsewhere.png',
    };
    const gallery = getGalleryStateView(resolvedValues(values), boards, null);

    expect(gallery.items).toEqual([]);
    expect(gallery.selectedItemKey).toBeNull();
  });

  it('parses persisted gallery settings with safe defaults', () => {
    const gallery = getGalleryStateView(
      resolvedValues({ boardOrderBy: 'board_name', mediaFilter: 'audio', starredSectionCollapsed: 'yes' }),
      boards,
      []
    );

    expect(gallery.settings).toEqual({
      autoAddBoardId: null,
      boardOrderBy: 'board_name',
      boardOrderDir: 'DESC',
      boardPanelCollapsed: false,
      boardPanelHeightPx: 280,
      boardPanelWidthPx: 240,
      collapsedBoardSections: [],
      collapsedProjectGroupIds: [],
      expandedBoardSections: [],
      followSelectedBoard: false,
      imageDensityPercent: 50,
      imageOrderDir: 'DESC',
      mediaFilter: 'all',
      middleClickOpensInNewTab: false,
      paginationMode: 'infinite',
      showArchivedBoards: false,
      showDateBoards: false,
      showImageDimensions: false,
      showOtherProjectBoards: false,
      showPendingItems: true,
      progressSectionCollapsed: false,
      searchScope: 'board',
      starredSectionCollapsed: false,
      thumbnailFit: 'square',
    });
    expect(
      getGalleryStateView(resolvedValues({ starredSectionCollapsed: true }), boards, []).settings
        .starredSectionCollapsed
    ).toBe(true);
    expect(getGalleryStateView(resolvedValues({ mediaFilter: 'video' }), boards, []).settings.mediaFilter).toBe(
      'video'
    );
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
        resolvedValues({ ...stamped, ...values, selectedImageQuery: { ...stamped.selectedImageQuery, ...stamp } }),
        boards,
        []
      ).revealTargetPage;

    expect(pageOf({})).toBe(2);
    expect(getGalleryStateView(resolvedValues(stamped), boards, []).page).toBe(0);
    expect(pageOf({ selectedBoardId: 'board-1' })).toBeNull();
    expect(pageOf({ paginationMode: 'infinite' })).toBeNull();
    expect(pageOf({}, { galleryView: 'assets' })).toBeNull();
    expect(pageOf({}, { paginationMode: 'infinite' })).toBeNull();
    expect(pageOf({ imageOrderDir: 'ASC' })).toBeNull();
    expect(pageOf({ searchTerm: 'cats' })).toBeNull();
    expect(pageOf({ starredOnly: true })).toBeNull();
    expect(pageOf({ starredOnly: true }, { starredOnly: true })).toBe(2);
    // A stamp without a media kind was made in a listing of both kinds, whatever the filter is now.
    expect(pageOf({ mediaFilter: 'video' })).toBeNull();
    expect(pageOf({ mediaFilter: 'video' }, { mediaKind: 'video' })).toBe(2);
    expect(pageOf({}, { mediaKind: 'image' })).toBeNull();
    // A starred selection sits in the strip, so no page of the unstarred grid holds it.
    expect(pageOf({ selectedImage: { ...createImageItem('starred.png'), starred: true } })).toBeNull();
    expect(
      pageOf(
        { selectedImage: { ...createImageItem('starred.png'), starred: true }, starredOnly: true },
        { starredOnly: true }
      )
    ).toBe(2);
    expect(pageOf({ semanticImageQuery: { imageName: 'ref.png', kind: 'image' } })).toBeNull();
    // A search of every board is its own listing: a stamp made there names it rather than the selected board.
    expect(pageOf({ searchScope: 'all', searchTerm: 'cats' }, { searchTerm: 'cats' })).toBeNull();
    expect(pageOf({ searchScope: 'all', searchTerm: 'cats' }, { boardId: 'all', searchTerm: 'cats' })).toBe(2);
    expect(pageOf({ searchTerm: 'cats' }, { boardId: 'all', searchTerm: 'cats' })).toBeNull();
  });

  it('qualifies legacy names and preserves ordered mixed-media selection keys', () => {
    const gallery = getGalleryStateView(
      resolvedValues({ selectedImageNames: ['a.png', 'video:shared', 'image:shared', 7] }),
      boards,
      []
    ) as ReturnType<typeof getGalleryStateView> & { selectedItemKeys?: string[] };

    expect(gallery.selectedItemKeys).toEqual(['image:a.png', 'video:shared', 'image:shared']);
    expect(
      (
        getGalleryStateView(resolvedValues({ selectedImageName: 'a.png' }), boards, []) as ReturnType<
          typeof getGalleryStateView
        > & {
          selectedItemKeys?: string[];
        }
      ).selectedItemKeys
    ).toEqual(['image:a.png']);
  });

  it('projects same-name images and videos independently by qualified key', () => {
    const image = createImageItem('shared');
    const video = createVideoItem('shared');
    const gallery = getGalleryStateView(
      resolvedValues({
        compareImage: image,
        selectedImage: video,
        selectedImageName: 'video:shared',
        selectedImageNames: ['image:shared', 'video:shared'],
      }),
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
});

describe('getGalleryListing', () => {
  const recent = createImageItem('just-generated.png');

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
});

describe('getGalleryDestinationBoardId', () => {
  it('sends results to the picked board, keeping an explicit Uncategorized choice', () => {
    expect(getGalleryDestinationBoardId({ projectBoardId: 'project', selectedBoardId: 'picked' })).toBe('picked');
    expect(getGalleryDestinationBoardId({ projectBoardId: 'project', selectedBoardId: 'none' })).toBe('none');
  });
});

describe('getGalleryAutoAddBoardId', () => {
  const anyBoard = () => true;

  it('sends results to the project inbox by default, wherever the gallery browses', () => {
    expect(
      getGalleryAutoAddBoardId(resolvedValues({ projectBoardId: 'project', selectedBoardId: 'picked' }), anyBoard)
    ).toBe('project');
    // What earlier builds stored for following the selected board, their default, reads as the default now.
    expect(
      getGalleryAutoAddBoardId(
        resolvedValues({ autoAddBoardId: 'follow', projectBoardId: 'project', selectedBoardId: 'picked' }),
        anyBoard
      )
    ).toBe('project');
    // A gallery with no inbox, such as a Library-only one, falls back to Uncategorized.
    expect(getGalleryAutoAddBoardId(resolvedValues({ selectedBoardId: 'picked' }), anyBoard)).toBe('none');
  });

  it('sends results a view-only board refuses to the project inbox, followed or fixed', () => {
    const viewOnly = (boardId: string) => boardId !== 'shared';

    expect(
      getGalleryAutoAddBoardId(
        resolvedValues({ followSelectedBoard: true, projectBoardId: 'inbox', selectedBoardId: 'shared' }),
        viewOnly
      )
    ).toBe('inbox');
    expect(
      getGalleryAutoAddBoardId(resolvedValues({ autoAddBoardId: 'shared', projectBoardId: 'inbox' }), viewOnly)
    ).toBe('inbox');
    expect(getGalleryAutoAddBoardId(resolvedValues({ autoAddBoardId: 'shared' }), viewOnly)).toBe('none');
    // Uncategorized is always the caller's own.
    expect(
      getGalleryAutoAddBoardId(resolvedValues({ autoAddBoardId: 'none', projectBoardId: 'inbox' }), () => false)
    ).toBe('none');
  });
});

describe('settleGalleryValues', () => {
  const selected = {
    selectedBoardId: 'dogs',
    selectedImage: { boardId: 'dogs', kind: 'image', name: 'a.png' },
    selectedImageName: 'image:a.png',
    selectedImageNames: ['image:a.png', 'image:b.png'],
  };

  it.each([
    ['another board', { selectedBoardId: 'cats' }],
    ['a search', { searchTerm: 'fox' }],
    ['the starred filter', { starredOnly: true }],
    ['the other tab', { galleryView: 'assets' }],
    ['a media filter', { mediaFilter: 'video' }],
    ['a semantic query', { semanticImageQuery: { kind: 'text', query: 'fox' }, semanticSearchText: 'fox' }],
  ])(
    'drops a selection the listing no longer holds after a switch to %s, keeping the item Preview shows',
    (_case, patch) => {
      const next = settleGalleryValues(selected, { ...selected, ...patch });

      expect(next.selectedImageNames).toEqual([]);
      expect(next.selectedImageName).toBe('image:a.png');
    }
  );

  it('drops a single legacy selection stored without its list', () => {
    const legacy = { selectedBoardId: 'dogs', selectedImageName: 'image:a.png' };

    expect(settleGalleryValues(legacy, { ...legacy, starredOnly: true }).selectedImageNames).toEqual([]);
  });
});
