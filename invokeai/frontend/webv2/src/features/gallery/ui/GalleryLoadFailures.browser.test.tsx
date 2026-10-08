/* oxlint-disable react-perf/jsx-no-new-object-as-prop */
import type { GalleryItem, GalleryItemsPage } from '@features/gallery/core/items';
import type { GalleryBoard, GeneratedImageContract } from '@features/gallery/core/types';
import type { QueueProgressSession } from '@features/queue/contracts';

import { Box, ChakraProvider } from '@chakra-ui/react';
import { DndContext } from '@dnd-kit/core';
import { GalleryItemActionsProvider, GalleryUiProvider, type GalleryUiAdapter } from '@features/gallery/react';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { createInstance } from 'i18next';
import { act, useMemo, useState, type ReactNode } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import { GalleryWidgetView } from './GalleryWidgetView';

interface ListRequest {
  boardId: string;
  offset: number;
  searchTerm: string;
  starred?: boolean;
}

/** The stubbed transport: each request is answered by whatever the test scripts for it. */
const transport = vi.hoisted(() => ({
  listBoards: vi.fn<() => Promise<GalleryBoard[]>>(),
  listItems: vi.fn<(request: ListRequest) => Promise<GalleryItemsPage>>(),
  listStarred: vi.fn<(request: ListRequest) => Promise<GalleryItemsPage>>(),
}));

vi.mock('@features/gallery/data/backend', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  fetchImageIndexAvailability: () => Promise.resolve({ modelName: null, state: 'disabled' }),
  listGalleryBoards: () => transport.listBoards(),
  listGalleryDateBoards: () => Promise.resolve([]),
  listGalleryItems: (request: ListRequest) =>
    request.starred === true ? transport.listStarred(request) : transport.listItems(request),
}));

vi.mock('@features/queue/react', () => ({
  useItemProgress: () => null,
  useQueueItemProgressImage: () => null,
}));

const i18n = createInstance();
await i18n.use(initReactI18next).init({
  interpolation: { escapeValue: false },
  lng: 'en',
  resources: { en: { translation: await fetch('/locales/en.json').then((response) => response.json()) } },
});
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const THUMBNAIL = `data:image/svg+xml,${encodeURIComponent(
  '<svg xmlns="http://www.w3.org/2000/svg" width="8" height="8"><rect width="8" height="8" fill="#547c83"/></svg>'
)}`;

const createBoard = (id: string, name: string): GalleryBoard => ({
  archived: false,
  assetCount: 0,
  assetVideoCount: 0,
  id,
  imageCount: 3,
  kind: 'board',
  name,
  isInbox: false,
  projectId: null,
  videoCount: 0,
});
const BOARDS = [
  createBoard('dogs', 'Dogs'),
  createBoard('cats', 'Cats'),
  { ...createBoard('none', ''), kind: 'uncategorized' as const },
];

const image = (name: string, boardId: string, day = 1): GalleryItem => ({
  boardId,
  category: 'general',
  createdAt: `2026-09-${String(day).padStart(2, '0')}T00:00:00.000Z`,
  fullUrl: THUMBNAIL,
  height: 8,
  isIntermediate: false,
  kind: 'image',
  name,
  starred: false,
  thumbnailUrl: THUMBNAIL,
  width: 8,
});
const page = (items: GalleryItem[], total = items.length): Promise<GalleryItemsPage> =>
  Promise.resolve({ items, total });
const fail = (): Promise<never> => Promise.reject(new Error('Service Unavailable'));

/** A just-generated image on another board, the kind the old fallback showed in place of a failed listing. */
const OTHER_BOARD_RECENT: GeneratedImageContract = {
  boardId: 'cats',
  height: 8,
  imageName: 'recent-elsewhere.png',
  imageUrl: THUMBNAIL,
  queuedAt: '2026-09-30T00:00:00.000Z',
  sourceQueueItemId: 'queue-1',
  thumbnailUrl: THUMBNAIL,
  width: 8,
};

const DOG_ITEMS = [image('a.png', 'dogs', 3), image('b.png', 'dogs', 2), image('c.png', 'dogs', 1)];
const CAT_ITEMS = [image('tabby.png', 'cats', 3)];

const noop = () => undefined;
const runtime = { commands: { register: () => noop }, hotkeys: { register: () => noop } };
const itemActions = {
  deleteItems: vi.fn(),
  downloadItem: vi.fn(),
  downloadItems: vi.fn(),
  moveItemsToBoard: vi.fn(),
  openItemInNewTab: vi.fn(),
  openItemInPreview: vi.fn(),
  setItemsStarred: vi.fn(),
};
const ItemActionsProvider = ({ children }: { children: ReactNode }) => (
  <GalleryItemActionsProvider actions={itemActions}>{children}</GalleryItemActionsProvider>
);

/** The workbench side: Gallery commands patch the values the widget reads back, as the real store does. */
const Harness = ({
  initialValues,
  progressSessions,
  region,
}: {
  initialValues: Record<string, unknown>;
  progressSessions: QueueProgressSession[];
  region: 'bottom' | 'center';
}) => {
  const [galleryValues, setGalleryValues] = useState(initialValues);
  const adapter = useMemo((): GalleryUiAdapter => {
    const patch = (values: Record<string, unknown>) => setGalleryValues((current) => ({ ...current, ...values }));

    return {
      ItemActionsProvider,
      ImageContextMenu: () => null,
      antialiasProgressImages: false,
      exportProject: noop,
      followProgressSession: noop,
      followedProgressSessionId: null,
      gallery: {
        clearSearch: () => patch({ searchTerm: '' }),
        clearSelection: noop,
        commitSemanticSearch: noop,
        reconcileDeletedBoardOutcome: noop,
        selectBoard: (selectedBoardId: string) => patch({ galleryPage: 0, selectedBoardId }),
        selectImage: noop,
        selectItem: (item: GalleryItem) => patch({ selectedImageName: `${item.kind}:${item.name}` }),
        setCompareImage: noop,
        setCompareItem: noop,
        setItemMultiSelection: noop,
        setPage: (galleryPage: number) => patch({ galleryPage }),
        setPageInfo: noop,
        setSearchTerm: (searchTerm: string) => patch({ galleryPage: 0, searchTerm }),
        setSemanticSearchMode: noop,
        setSemanticSearchText: noop,
        setStarredOnly: (starredOnly: boolean) => patch({ starredOnly }),
        setView: (galleryView: string) => patch({ galleryView }),
        toggleItemSelection: noop,
        updateSettings: (settings: Record<string, unknown>) => patch(settings),
      },
      galleryValues,
      generateValues: {},
      getItemLabel: () => Promise.resolve(null),
      liveFollowEnabled: false,
      notifications: { add: noop, reportError: noop },
      pinnedProgressSessionId: null,
      progressSessions,
      projectId: 'project-1',
      projectName: 'Project',
      widgets: { openGallery: () => true, patchGalleryValues: patch },
    };
  }, [galleryValues, progressSessions]);

  return (
    <GalleryUiProvider adapter={adapter}>
      <DndContext>
        <Box h="520px" w="960px">
          <GalleryWidgetView presentation="compact" region={region} runtime={runtime} />
        </Box>
      </DndContext>
    </GalleryUiProvider>
  );
};

let host: HTMLDivElement | null = null;
let root: Root | null = null;
let queryClient: QueryClient | null = null;

const NO_SESSIONS: QueueProgressSession[] = [];
const PROGRESS_SESSION: QueueProgressSession = {
  backendItemId: 10,
  height: 768,
  id: 'running',
  itemCount: 1,
  itemIndex: 1,
  label: 'Run running',
  queueItemId: 'running',
  sourceId: 'generate',
  state: 'running',
  width: 512,
};

const renderGallery = async (
  values: Record<string, unknown> = {},
  region: 'bottom' | 'center' = 'center',
  progressSessions = NO_SESSIONS
) => {
  await act(() =>
    root?.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <QueryClientProvider client={queryClient!}>
            <Harness
              initialValues={{ selectedBoardId: 'dogs', ...values }}
              progressSessions={progressSessions}
              region={region}
            />
          </QueryClientProvider>
        </ChakraProvider>
      </I18nextProvider>
    )
  );
};

const listing = () => host?.querySelector('[role="list"][aria-label="Gallery items"]') ?? null;
const thumbnailNames = () =>
  [...(listing()?.querySelectorAll<HTMLElement>('[role="listitem"] button[aria-pressed]') ?? [])].map(
    (button) => button.getAttribute('aria-label') ?? ''
  );
const alertText = () => host?.querySelector('[role="alert"]')?.textContent ?? null;
const findButton = (name: string) => host?.querySelector<HTMLButtonElement>(`button[aria-label="${name}"]`) ?? null;
const waitFor = (assertion: () => void) => vi.waitFor(assertion, { timeout: 5_000 });
const announcer = () => host?.querySelector('[role="status"][aria-live="polite"][aria-atomic="true"]') ?? null;
const gridViewport = () => listing()?.closest<HTMLElement>('[data-part="viewport"]') ?? null;
/** The compact chip's text; the drag context adds hidden instructions of its own to the host. */
const chipText = () => host?.querySelector('p')?.textContent ?? null;
const offsetsRequested = () => transport.listItems.mock.calls.map(([request]) => request.offset);
/** Lets frames, effects and any request they would start run their course. */
const settleFrames = async (frames = 8) => {
  for (let frame = 0; frame < frames; frame += 1) {
    await act(
      () =>
        new Promise<void>((resolve) => {
          requestAnimationFrame(() => setTimeout(resolve, 10));
        })
    );
  }
};
/** Each a minute older than the last, so the grid shows them in this order, page after page. */
const dogs = (count: number, from = 0) =>
  Array.from({ length: count }, (_, index) => ({
    ...image(`dog-${String(from + index).padStart(3, '0')}.png`, 'dogs'),
    createdAt: new Date(Date.UTC(2026, 8, 30) - (from + index) * 60_000).toISOString(),
  }));

/**
 * Scrolls the grid to its end, which requests the next page; once that has failed, settles at the new end and
 * measures the failure notice against the viewport and the loaded tiles.
 */
const scrollToLoadMoreFailure = async () => {
  const viewport = gridViewport()!;
  const scrollToEnd = () => {
    viewport.scrollTop = viewport.scrollHeight;
    viewport.dispatchEvent(new Event('scroll'));
  };

  await act(scrollToEnd);
  await waitFor(() => expect(findButton('Retry loading more items')).not.toBeNull());
  await act(scrollToEnd);
  // Outlast the virtualizer's 150ms scrolling state, as a user reading the notice would.
  await act(
    () =>
      new Promise<void>((resolve) => {
        setTimeout(resolve, 200);
      })
  );
  expect(viewport.scrollTop + viewport.clientHeight).toBeGreaterThanOrEqual(viewport.scrollHeight - 1);

  const rect = findButton('Retry loading more items')!.parentElement!.getBoundingClientRect();
  const viewportRect = viewport.getBoundingClientRect();
  const tiles = [...listing()!.querySelectorAll<HTMLElement>('[role="listitem"]')];
  // The end of the bottom row; tiles of one row may differ by a subpixel.
  const lastTile = tiles.reduce((last, tile) =>
    tile.getBoundingClientRect().bottom > last.getBoundingClientRect().bottom - 1 ? tile : last
  );

  return {
    inViewport: rect.top >= viewportRect.top - 1 && rect.bottom <= viewportRect.bottom + 1,
    lastTileBottom: lastTile.getBoundingClientRect().bottom,
    lastTileName: lastTile.querySelector('button[aria-pressed]')?.getAttribute('aria-label'),
    overlappingTiles: tiles.filter((tile) => {
      const tileRect = tile.getBoundingClientRect();

      return tileRect.top < rect.bottom && tileRect.bottom > rect.top;
    }).length,
    rect,
  };
};

beforeEach(() => {
  accountLifecycle.activate('gallery-load-failures');
  vi.clearAllMocks();
  transport.listBoards.mockImplementation(() => Promise.resolve(BOARDS));
  transport.listItems.mockImplementation(({ boardId }) => page(boardId === 'cats' ? CAT_ITEMS : DOG_ITEMS));
  transport.listStarred.mockImplementation(() => page([]));
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
});

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  queryClient?.clear();
  host = null;
  root = null;
  queryClient = null;
});

describe('Gallery listing failures', () => {
  it('reports a failed first load instead of an empty board, then recovers with Retry and keeps the selection', async () => {
    transport.listItems.mockImplementationOnce(fail);
    await renderGallery({ selectedImageName: 'image:b.png' });

    await waitFor(() => expect(alertText()).toContain('Could not load these items.'));
    expect(alertText()).toContain('Service Unavailable');
    expect(host?.textContent).not.toContain('Drop media here or click to upload');
    expect(host?.textContent).not.toContain('No items match');
    // The board list is independent of the listing and still loads.
    expect(host?.textContent).toContain('Cats');

    const retry = findButton('Retry loading items');

    retry?.focus();
    await userEvent.keyboard('{Enter}');

    await waitFor(() => expect(thumbnailNames()).toHaveLength(3));
    expect(alertText()).toBeNull();
    expect(
      listing()?.querySelector('button[aria-label="Select b.png for preview"]')?.getAttribute('aria-current')
    ).toBe('true');
    // Focus went to the grid that replaced the Retry, not the document body.
    await waitFor(() => expect(host?.contains(document.activeElement)).toBe(true));
  });

  it('shows an anchored window that failed as failed, never the unfiltered recents', async () => {
    transport.listItems.mockImplementation(fail);
    await renderGallery({ galleryPage: 2, paginationMode: 'paginated', recentImages: [OTHER_BOARD_RECENT] });

    await waitFor(() => expect(alertText()).toContain('Could not load these items.'));
    expect(host?.textContent).not.toContain('recent-elsewhere.png');
    expect(thumbnailNames()).toEqual([]);
  });

  it("shows a failed board switch as the new board's failure, not the previous board's items", async () => {
    await renderGallery();
    await waitFor(() => expect(thumbnailNames()).toHaveLength(3));

    transport.listItems.mockImplementationOnce(fail);
    const catsRow = [...(host?.querySelectorAll<HTMLButtonElement>('button') ?? [])].find((button) =>
      button.textContent?.startsWith('Cats')
    );

    await act(() => catsRow?.click());

    await waitFor(() => expect(alertText()).toContain('Could not load these items.'));
    expect(thumbnailNames()).toEqual([]);
    expect(host?.textContent).not.toContain('a.png');

    await act(() => findButton('Retry loading items')?.click());

    await waitFor(() => expect(thumbnailNames()).toEqual(['Select tabby.png for preview']));
    expect(transport.listItems).toHaveBeenLastCalledWith(expect.objectContaining({ boardId: 'cats' }));
  });

  it('keeps loaded pages when the next page fails, offers Retry there, and substitutes no recents', async () => {
    // A short first page of a long board: the end of the grid is in view, so the next page is requested at once.
    transport.listItems.mockImplementation(({ offset }) => (offset === 0 ? page(DOG_ITEMS, 500) : fail()));
    await renderGallery({ recentImages: [OTHER_BOARD_RECENT] });

    await waitFor(() => expect(host?.textContent).toContain('Could not load more items.'));
    expect(thumbnailNames()).toEqual([
      'Select a.png for preview',
      'Select b.png for preview',
      'Select c.png for preview',
    ]);
    expect(host?.textContent).not.toContain('recent-elsewhere.png');
    expect(alertText()).toBeNull();

    // The failed page waits for Retry: the end of the grid stays in view, yet nothing asks for it again.
    const failedCalls = transport.listItems.mock.calls.length;

    await settleFrames();
    expect(transport.listItems.mock.calls.length).toBe(failedCalls);

    transport.listItems.mockImplementation(({ offset }) =>
      offset === 0 ? page(DOG_ITEMS, 4) : page([image('d.png', 'dogs', 0)], 4)
    );
    await act(() => findButton('Retry loading more items')?.click());

    await waitFor(() => expect(thumbnailNames()).toHaveLength(4));
    expect(host?.textContent).not.toContain('Could not load more items.');
    // Retry fetched only the failed page, not the whole listing again.
    expect(transport.listItems.mock.calls.slice(failedCalls).map(([request]) => request.offset)).toEqual([60]);
  });

  it("keeps a scope's earlier results through a failed refresh, with a notice that Retry clears", async () => {
    await renderGallery();
    await waitFor(() => expect(thumbnailNames()).toHaveLength(3));

    // The live region exists, empty, before anything fails, so the failure is a change inside it.
    const liveRegion = announcer();

    expect(liveRegion?.textContent).toBe('');

    transport.listItems.mockImplementation(fail);
    await act(() => queryClient!.invalidateQueries({ queryKey: ['gallery', 'items', 'list'] }));

    await waitFor(() => expect(host?.textContent).toContain('Could not refresh. Showing earlier results.'));
    expect(thumbnailNames()).toHaveLength(3);
    expect(alertText()).toBeNull();
    expect(announcer()).toBe(liveRegion);
    expect(liveRegion?.textContent).toBe('Could not refresh. Showing earlier results.');
    // Announced once, through the persistent region only.
    expect(
      [...(host?.querySelectorAll('[role="status"], [role="alert"], [aria-live]') ?? [])].filter((region) =>
        region.textContent?.includes('Could not refresh.')
      )
    ).toEqual([liveRegion]);

    transport.listItems.mockImplementation(() => page([image('fresh.png', 'dogs', 9), ...DOG_ITEMS]));
    await act(() => findButton('Retry loading items')?.click());

    await waitFor(() => expect(thumbnailNames()).toHaveLength(4));
    expect(host?.textContent).not.toContain('Could not refresh.');
  });

  it('keeps a failed next page failed through an unrelated refetch, without retrying it on its own', async () => {
    transport.listItems.mockImplementation(({ offset }) => (offset === 0 ? page(DOG_ITEMS, 500) : fail()));
    await renderGallery();
    await waitFor(() => expect(host?.textContent).toContain('Could not load more items.'));

    const failedPageRequests = offsetsRequested().filter((offset) => offset === 60).length;

    // An invalidation, as after a generation, refetches the loaded page successfully.
    await act(() => queryClient!.invalidateQueries({ queryKey: ['gallery', 'items', 'list'] }));
    await waitFor(() => expect(offsetsRequested().filter((offset) => offset === 0)).toHaveLength(2));
    await settleFrames();

    expect(offsetsRequested().filter((offset) => offset === 60)).toHaveLength(failedPageRequests);
    expect(host?.textContent).toContain('Could not load more items.');
    expect(host?.textContent).not.toContain('Could not refresh.');
    expect(findButton('Retry loading more items')?.getAttribute('aria-busy')).toBeNull();
  });

  it('shows the load-more failure after the last loaded row, in view at the end of the grid', async () => {
    transport.listItems.mockImplementation(({ offset }) => (offset === 0 ? page(dogs(60), 500) : fail()));
    await renderGallery();
    await waitFor(() => expect(thumbnailNames().length).toBeGreaterThan(0));

    const viewport = gridViewport()!;
    const notice = await scrollToLoadMoreFailure();

    expect(notice.inViewport).toBe(true);
    expect(notice.overlappingTiles).toBe(0);
    // The last of the 60 loaded tiles sits directly above it.
    expect(notice.lastTileName).toBe('Select dog-059.png for preview');
    expect(notice.rect.top).toBeGreaterThanOrEqual(notice.lastTileBottom);
    // The listing's box spans its rows, so whatever follows it in flow cannot land among them.
    expect(listing()!.getBoundingClientRect().bottom).toBeGreaterThanOrEqual(notice.lastTileBottom);

    // Keyboard users reach Retry from the last tile.
    const lastTile = listing()!.querySelector<HTMLElement>('button[aria-label="Select dog-059.png for preview"]')!;

    lastTile.focus({ preventScroll: true });
    await userEvent.tab();
    expect(document.activeElement?.getAttribute('aria-label')).toBe('Retry loading more items');

    transport.listItems.mockImplementation(({ offset }) => page(dogs(60, offset), 500));
    await userEvent.keyboard('{Enter}');
    await waitFor(() => expect(findButton('Retry loading more items')).toBeNull());
    await settleFrames(2);

    // The next page's first tiles take the notice's place, where the user is looking.
    const viewportRect = viewport.getBoundingClientRect();
    const tileInPlace = [...listing()!.querySelectorAll<HTMLElement>('[role="listitem"]')].find((tile) => {
      const tileRect = tile.getBoundingClientRect();

      return tileRect.top < notice.rect.bottom && tileRect.bottom > notice.rect.top;
    });

    expect(tileInPlace?.querySelector('button[aria-pressed]')?.getAttribute('aria-label')).toMatch(
      /^Select dog-0(6\d)\.png for preview$/
    );
    expect(tileInPlace!.getBoundingClientRect().top).toBeLessThan(viewportRect.bottom);
  });

  it('places the load-more failure after the rows below the starred strip and progress tiles, at any density', async () => {
    for (const imageDensityPercent of [0, 100]) {
      transport.listItems.mockImplementation(({ offset }) => (offset === 0 ? page(dogs(60), 500) : fail()));
      transport.listStarred.mockImplementation(() =>
        page(Array.from({ length: 4 }, (_, index) => ({ ...image(`fav-${index}.png`, 'dogs', 9), starred: true })))
      );
      await renderGallery({ imageDensityPercent }, 'center', [PROGRESS_SESSION]);
      await waitFor(() => expect(host?.querySelector('[role="list"][aria-label="Starred"]')).not.toBeNull());
      expect(host?.querySelector('[data-gallery-session-id]')).not.toBeNull();

      const notice = await scrollToLoadMoreFailure();

      expect(notice.inViewport, `density ${imageDensityPercent}`).toBe(true);
      expect(notice.overlappingTiles, `density ${imageDensityPercent}`).toBe(0);
      expect(notice.lastTileName, `density ${imageDensityPercent}`).toBe('Select dog-059.png for preview');
      expect(notice.rect.top, `density ${imageDensityPercent}`).toBeGreaterThanOrEqual(notice.lastTileBottom);

      await act(() => root?.render(null));
      queryClient?.clear();
    }
  });

  it('hands focus to the grid beside the new items after a load-more Retry, without scrolling', async () => {
    transport.listItems.mockImplementation(({ offset }) => (offset === 0 ? page(dogs(60), 500) : fail()));
    await renderGallery();
    await waitFor(() => expect(thumbnailNames().length).toBeGreaterThan(0));

    const viewport = gridViewport()!;

    await scrollToLoadMoreFailure();

    const retry = findButton('Retry loading more items')!;
    const scrollTop = viewport.scrollTop;

    expect(scrollTop).toBeGreaterThan(0);
    let deliverNextPage: () => void = noop;

    transport.listItems.mockImplementation(
      ({ offset }) =>
        new Promise((resolve) => {
          deliverNextPage = () => resolve({ items: dogs(60, offset), total: 500 });
        })
    );
    // A real click, which focuses the button as a user's would; it keeps focus while the retry runs.
    await userEvent.click(retry);
    expect(document.activeElement).toBe(retry);
    expect(retry.getAttribute('aria-busy')).toBe('true');

    await act(() => deliverNextPage());
    await waitFor(() => expect(findButton('Retry loading more items')).toBeNull());
    await settleFrames(2);

    expect(Math.abs(viewport.scrollTop - scrollTop)).toBeLessThanOrEqual(1);

    const focused = document.activeElement as HTMLElement;
    const focusedRect = focused.getBoundingClientRect();
    const viewportRect = viewport.getBoundingClientRect();

    expect(focused.matches('[role="listitem"] button[aria-pressed]')).toBe(true);
    expect(focusedRect.bottom).toBeGreaterThan(viewportRect.top);
    expect(focusedRect.top).toBeLessThan(viewportRect.bottom);
    // Beside the row that was retried, not back at the top of the board.
    expect(focused.getAttribute('aria-label')).not.toBe('Select dog-000.png for preview');
  });
});

describe('Gallery compact status chip', () => {
  it('states no count it does not know, and says when the listing could not load', async () => {
    transport.listItems.mockImplementation(() => new Promise(noop));
    await renderGallery({}, 'bottom');
    await settleFrames(2);

    expect(chipText()).toBe('Gallery');

    await act(() => root?.unmount());
    root = createRoot(host!);
    queryClient!.clear();
    transport.listItems.mockImplementation(fail);
    await renderGallery({}, 'bottom');

    await waitFor(() => expect(chipText()).toBe('Gallery: could not load'));

    transport.listItems.mockImplementation(() => page(DOG_ITEMS, 42));
    await act(() => queryClient!.refetchQueries({ queryKey: ['gallery', 'items', 'list'] }));

    await waitFor(() => expect(chipText()).toBe('Gallery: 42 items'));
  });
});

describe('Gallery side-section failures', () => {
  it('reports a failed board list in the board panel without claiming there are none or blocking the grid', async () => {
    transport.listBoards.mockImplementationOnce(fail);
    await renderGallery();

    await waitFor(() => expect(host?.textContent).toContain('Could not load boards.'));
    await waitFor(() => expect(thumbnailNames()).toHaveLength(3));
    expect(host?.textContent).not.toContain('No boards match');
    expect(alertText()).toBeNull();

    await act(() => findButton('Retry loading boards')?.click());

    await waitFor(() => expect(host?.textContent).toContain('Cats'));
    expect(host?.textContent).not.toContain('Could not load boards.');
  });

  it('reports a failed starred strip in its place without blocking the grid', async () => {
    transport.listStarred.mockImplementationOnce(fail);
    await renderGallery();

    await waitFor(() => expect(host?.textContent).toContain('Could not load starred items.'));
    expect(thumbnailNames()).toHaveLength(3);

    transport.listStarred.mockImplementation(() => page([{ ...image('fav.png', 'dogs', 9), starred: true }]));
    await act(() => findButton('Retry loading starred items')?.click());

    await waitFor(() => expect(host?.textContent).not.toContain('Could not load starred items.'));
    expect(
      host?.querySelector('[role="list"][aria-label="Starred"] button[aria-label="Select fav.png for preview"]')
    ).not.toBeNull();
  });

  it('reports a failed refresh of an empty strip, and claims no empty board while starred items are unknown', async () => {
    transport.listItems.mockImplementation(() => page([]));
    await renderGallery();
    await waitFor(() => expect(host?.textContent).toContain('Drop media here or click to upload'));

    transport.listStarred.mockImplementation(fail);
    await act(() => queryClient!.invalidateQueries({ queryKey: ['gallery', 'items', 'list'] }));

    await waitFor(() => expect(host?.textContent).toContain('Could not refresh starred items.'));
    expect(findButton('Retry refreshing starred items')).not.toBeNull();
    expect(host?.textContent).not.toContain('Drop media here or click to upload');
    expect(host?.textContent).not.toContain('No items match');

    transport.listStarred.mockImplementation(() => page([]));
    await act(() => findButton('Retry refreshing starred items')?.click());

    await waitFor(() => expect(host?.textContent).toContain('Drop media here or click to upload'));
    expect(host?.textContent).not.toContain('Could not refresh starred items.');
  });
});
