import { act, useEffect } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const mocks = vi.hoisted(() => ({
  fetchQuery: vi.fn(),
  galleryValues: {} as Record<string, unknown>,
  patchValues: vi.fn(),
  prefetchInfiniteQuery: vi.fn(),
  registerImageCluster: vi.fn(),
  resolveMany: vi.fn(),
  selectBoard: vi.fn(),
  selectItem: vi.fn(),
  setPage: vi.fn(),
  settings: { imageOrderDir: 'DESC', paginationMode: 'paginated', starredFirst: true } as Record<string, unknown>,
  setView: vi.fn(),
}));

vi.mock('@features/gallery', () => ({
  galleryImages: { resolveMany: mocks.resolveMany },
  legacyGeneratedImageToGalleryItem: (image: { image_name: string }) => image,
}));

vi.mock('@features/gallery/contracts', () => ({
  getGallerySettings: () => mocks.settings,
  registerImageCluster: mocks.registerImageCluster,
}));

vi.mock('@features/gallery/queries', () => ({
  GALLERY_MAX_ROWS: 600,
  GALLERY_PAGE_SIZE: 60,
  galleryItemNamesOptions: (filter: unknown) => ({ filter, queryKey: ['names', filter] }),
  galleryItemsInfiniteOptions: (filter: unknown, window: unknown) => ({
    filter,
    queryKey: ['items', filter, window],
    window,
  }),
}));

vi.mock('@tanstack/react-query', () => ({
  useQueryClient: () => ({ fetchQuery: mocks.fetchQuery, prefetchInfiniteQuery: mocks.prefetchInfiniteQuery }),
}));

vi.mock('@workbench/widgetState', () => ({
  getProjectWidgetValues: () => mocks.galleryValues,
}));

vi.mock('@workbench/WorkbenchContext', () => ({
  useWorkbenchCommands: () => ({
    gallery: {
      selectBoard: mocks.selectBoard,
      selectItem: mocks.selectItem,
      setPage: mocks.setPage,
      setView: mocks.setView,
    },
    widgets: { patchValues: mocks.patchValues },
  }),
  useWorkbenchQueries: () => ({ getSnapshot: () => ({ activeProject: { id: 'project-1' } }) }),
}));

import { useMapSelection } from './useSelectMapImage';

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let host: HTMLDivElement | null = null;
let root: Root | null = null;

/** Published from an effect, not during render, so the probe stays render-pure. */
const handle: {
  click: ((imageName: string) => void) | null;
  clickCluster: ((primaryImageName: string, imageNames: string[], label: string) => void) | null;
} = { click: null, clickCluster: null };

const Probe = () => {
  const { selectCluster, selectImage } = useMapSelection();

  useEffect(() => {
    handle.click = selectImage;
    handle.clickCluster = selectCluster;
  }, [selectCluster, selectImage]);

  return null;
};

/** Runs `fn` inside act and drains queued promise callbacks (a macrotask covers the chained awaits). */
const flush = async (fn: () => void = () => {}) => {
  await act(async () => {
    fn();
    await new Promise((resolve) => {
      setTimeout(resolve, 0);
    });
  });
};

const mount = async () => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  await flush(() => root?.render(<Probe />));
};

const unmount = async () => {
  await flush(() => root?.unmount());
  host?.remove();
  root = null;
  host = null;
  handle.click = null;
  handle.clickCluster = null;
};

/** A promise plus the trigger that settles it, so click ordering can be forced. */
const deferred = <T,>() => {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((r) => {
    resolve = r;
  });

  return { promise, resolve };
};

/** A names response placing `imageName` at `index` in its board's ordering. */
const namesWithImageAt = (imageName: string, index: number) => ({
  items: Array.from({ length: index + 1 }, (_, position) => ({
    kind: 'image',
    name: position === index ? imageName : `other-${String(position)}.png`,
  })),
  starredCount: 0,
  total: index + 1,
});

beforeEach(() => {
  mocks.galleryValues = {};
  mocks.settings = { imageOrderDir: 'DESC', paginationMode: 'paginated', starredFirst: true };
  mocks.fetchQuery.mockResolvedValue({ items: [], starredCount: 0, total: 0 });
  mocks.registerImageCluster.mockReturnValue('cluster-key-1');
});

afterEach(async () => {
  if (root) {
    await unmount();
  }
  mocks.fetchQuery.mockReset();
  mocks.patchValues.mockReset();
  mocks.prefetchInfiniteQuery.mockReset();
  mocks.registerImageCluster.mockReset();
  mocks.resolveMany.mockReset();
  mocks.selectBoard.mockReset();
  mocks.selectItem.mockReset();
  mocks.setPage.mockReset();
  mocks.setView.mockReset();
});

describe('useMapSelection', () => {
  describe('selectImage', () => {
    it('dispatches the selection for a click', async () => {
      mocks.resolveMany.mockResolvedValue([{ boardId: 'board-a', image_name: 'a.png', imageCategory: 'general' }]);
      await mount();

      await flush(() => handle.click?.('a.png'));

      expect(mocks.selectItem).toHaveBeenCalledTimes(1);
      expect(mocks.selectItem.mock.calls[0]?.[0]).toEqual({
        boardId: 'board-a',
        image_name: 'a.png',
        imageCategory: 'general',
      });
    });

    it("selects the image's board before the image itself", async () => {
      // The map spans every accessible board, but selectGalleryItem stamps the
      // navigation query from whatever list the gallery is currently showing. A
      // cross-board click without this left that query describing a list the
      // image was never in, and Preview's next/prev found no cursor and went
      // dead until the user re-selected from the grid.
      mocks.resolveMany.mockResolvedValue([
        { boardId: 'board-portraits', image_name: 'a.png', imageCategory: 'general' },
      ]);
      await mount();

      await flush(() => handle.click?.('a.png'));

      expect(mocks.selectBoard).toHaveBeenCalledWith('board-portraits');
      expect(mocks.selectBoard.mock.invocationCallOrder[0]).toBeLessThan(mocks.selectItem.mock.invocationCallOrder[0]);
    });

    it('lands the gallery on the page holding the image in paginated mode', async () => {
      mocks.settings = { imageOrderDir: 'DESC', paginationMode: 'paginated', starredFirst: true };
      mocks.resolveMany.mockResolvedValue([{ boardId: 'board-a', image_name: 'deep.png', imageCategory: 'general' }]);
      mocks.fetchQuery.mockResolvedValue(namesWithImageAt('deep.png', 130));
      await mount();

      await flush(() => handle.click?.('deep.png'));

      expect(mocks.setPage).toHaveBeenCalledWith(2);
      expect(mocks.selectItem).toHaveBeenCalledTimes(1);
      // The selection page rides along so the stamped navigation query
      // describes the page the image is actually on.
      expect(mocks.selectItem.mock.calls[0]?.[2]).toBe(2);
    });

    it("resolves the image's position against the listing the reveal lands on", async () => {
      mocks.settings = { imageOrderDir: 'ASC', paginationMode: 'paginated', starredFirst: false };
      mocks.resolveMany.mockResolvedValue([{ boardId: 'board-a', image_name: 'a.png', imageCategory: 'general' }]);
      mocks.fetchQuery.mockResolvedValue(namesWithImageAt('a.png', 0));
      await mount();

      await flush(() => handle.click?.('a.png'));

      expect(mocks.fetchQuery.mock.calls[0]?.[0].filter).toEqual({
        boardId: 'board-a',
        galleryView: 'images',
        orderDir: 'ASC',
        searchTerm: '',
        starredFirst: false,
      });
    });

    it('prefetches the pages down to the image in infinite mode', async () => {
      mocks.settings = { imageOrderDir: 'DESC', paginationMode: 'infinite', starredFirst: true };
      mocks.resolveMany.mockResolvedValue([{ boardId: 'board-a', image_name: 'deep.png', imageCategory: 'general' }]);
      mocks.fetchQuery.mockResolvedValue(namesWithImageAt('deep.png', 130));
      await mount();

      await flush(() => handle.click?.('deep.png'));

      expect(mocks.setPage).not.toHaveBeenCalled();
      expect(mocks.prefetchInfiniteQuery).toHaveBeenCalledTimes(1);
      expect(mocks.prefetchInfiniteQuery.mock.calls[0]?.[0].pages).toBe(3);
      expect(mocks.selectItem.mock.calls[0]?.[2]).toBe(2);
    });

    it('does not prefetch past the infinite window cap', async () => {
      // The infinite window cannot reach beyond GALLERY_MAX_ROWS, so loading
      // pages toward an unreachable image would only burn requests. The
      // selection itself still lands (Preview follows it).
      mocks.settings = { imageOrderDir: 'DESC', paginationMode: 'infinite', starredFirst: true };
      mocks.resolveMany.mockResolvedValue([{ boardId: 'board-a', image_name: 'deep.png', imageCategory: 'general' }]);
      mocks.fetchQuery.mockResolvedValue(namesWithImageAt('deep.png', 700));
      await mount();

      await flush(() => handle.click?.('deep.png'));

      expect(mocks.prefetchInfiniteQuery).not.toHaveBeenCalled();
      expect(mocks.selectItem).toHaveBeenCalledTimes(1);
    });

    it('clears an active search and similarity filter before revealing', async () => {
      // The image was located in the plain board listing; an active filter
      // would show some other list entirely.
      mocks.galleryValues = { searchTerm: 'sunset', semanticImageQuery: { kind: 'text', query: 'sunset' } };
      mocks.resolveMany.mockResolvedValue([{ boardId: 'board-a', image_name: 'a.png', imageCategory: 'general' }]);
      await mount();

      await flush(() => handle.click?.('a.png'));

      expect(mocks.patchValues).toHaveBeenCalledWith('gallery', { searchTerm: '', semanticImageQuery: null });
    });

    it('leaves the filters alone when none are active', async () => {
      mocks.resolveMany.mockResolvedValue([{ boardId: 'board-a', image_name: 'a.png', imageCategory: 'general' }]);
      await mount();

      await flush(() => handle.click?.('a.png'));

      expect(mocks.patchValues).not.toHaveBeenCalled();
    });

    it('switches the gallery to the assets tab for a non-general image', async () => {
      mocks.resolveMany.mockResolvedValue([{ boardId: 'board-a', image_name: 'a.png', imageCategory: 'user' }]);
      await mount();

      await flush(() => handle.click?.('a.png'));

      expect(mocks.setView).toHaveBeenCalledWith('assets');
    });

    it('does not touch the view when it already matches', async () => {
      mocks.resolveMany.mockResolvedValue([{ boardId: 'board-a', image_name: 'a.png', imageCategory: 'general' }]);
      await mount();

      await flush(() => handle.click?.('a.png'));

      expect(mocks.setView).not.toHaveBeenCalled();
    });

    it('still selects when the position lookup fails', async () => {
      mocks.resolveMany.mockResolvedValue([{ boardId: 'board-a', image_name: 'a.png', imageCategory: 'general' }]);
      mocks.fetchQuery.mockRejectedValue(new Error('names endpoint down'));
      await mount();

      await flush(() => handle.click?.('a.png'));

      expect(mocks.setPage).not.toHaveBeenCalled();
      expect(mocks.selectBoard).toHaveBeenCalledWith('board-a');
      expect(mocks.selectItem).toHaveBeenCalledTimes(1);
      expect(mocks.selectItem.mock.calls[0]?.[2]).toBeUndefined();
    });

    it('does not touch the board for a click that never resolves an image', async () => {
      mocks.resolveMany.mockResolvedValue([]);
      await mount();

      await flush(() => handle.click?.('gone.png'));

      expect(mocks.selectBoard).not.toHaveBeenCalled();
      expect(mocks.selectItem).not.toHaveBeenCalled();
    });
  });

  describe('selectCluster', () => {
    it('shows the cluster as a gallery filter with the clicked image selected', async () => {
      mocks.resolveMany.mockResolvedValue([{ boardId: 'board-a', image_name: 'a.png', imageCategory: 'general' }]);
      await mount();

      await flush(() => handle.clickCluster?.('a.png', ['a.png', 'b.png', 'c.png'], 'beaches'));

      expect(mocks.registerImageCluster).toHaveBeenCalledWith(['a.png', 'b.png', 'c.png'], 'beaches');
      expect(mocks.patchValues).toHaveBeenCalledWith('gallery', {
        galleryPage: 0,
        searchTerm: '',
        semanticImageQuery: { clusterId: 'cluster-key-1', kind: 'cluster', label: 'beaches' },
      });
      expect(mocks.selectItem).toHaveBeenCalledTimes(1);
      expect(mocks.selectItem.mock.calls[0]?.[0]).toEqual({
        boardId: 'board-a',
        image_name: 'a.png',
        imageCategory: 'general',
      });
    });

    it("selects the primary image's board before the cluster filter", async () => {
      // The selection stamps the navigation query from the list the gallery
      // is currently showing, so the primary image's board must be current
      // before the selection lands.
      mocks.resolveMany.mockResolvedValue([
        { boardId: 'board-landscapes', image_name: 'a.png', imageCategory: 'general' },
      ]);
      await mount();

      await flush(() => handle.clickCluster?.('a.png', ['a.png', 'b.png'], 'label'));

      expect(mocks.selectBoard).toHaveBeenCalledWith('board-landscapes');
      expect(mocks.selectBoard.mock.invocationCallOrder[0]).toBeLessThan(mocks.selectItem.mock.invocationCallOrder[0]);
    });

    it('leaves the gallery alone when the primary image cannot be resolved', async () => {
      mocks.resolveMany.mockResolvedValue([]);
      await mount();

      await flush(() => handle.clickCluster?.('gone.png', ['gone.png', 'b.png'], 'label'));

      expect(mocks.registerImageCluster).not.toHaveBeenCalled();
      expect(mocks.patchValues).not.toHaveBeenCalled();
      expect(mocks.selectItem).not.toHaveBeenCalled();
    });
  });

  it('shares one sequence guard across both modes, so the newer click wins', async () => {
    // The two entry points must not race each other: switching cluster mode
    // mid-flight would otherwise let a stale resolution overwrite a newer
    // selection.
    const slow = deferred<{ boardId: string; image_name: string; imageCategory: string }[]>();
    const fast = deferred<{ boardId: string; image_name: string; imageCategory: string }[]>();

    mocks.resolveMany.mockReturnValueOnce(slow.promise).mockReturnValueOnce(fast.promise);
    await mount();

    await flush(() => {
      handle.clickCluster?.('slow.png', ['slow.png'], 'label');
      handle.click?.('fast.png');
    });
    await flush(() => {
      fast.resolve([{ boardId: 'board-a', image_name: 'fast.png', imageCategory: 'general' }]);
      slow.resolve([{ boardId: 'board-a', image_name: 'slow.png', imageCategory: 'general' }]);
    });

    expect(mocks.patchValues).not.toHaveBeenCalled();
    expect(mocks.selectItem).toHaveBeenCalledTimes(1);
    expect(mocks.selectItem.mock.calls[0]?.[0].image_name).toBe('fast.png');
  });

  it('ignores a slow click that resolves after a newer one', async () => {
    const slow = deferred<{ boardId: string; image_name: string; imageCategory: string }[]>();
    const fast = deferred<{ boardId: string; image_name: string; imageCategory: string }[]>();

    mocks.resolveMany.mockReturnValueOnce(slow.promise).mockReturnValueOnce(fast.promise);
    await mount();

    await flush(() => {
      handle.click?.('slow.png');
      handle.click?.('fast.png');
    });

    await flush(() => {
      fast.resolve([{ boardId: 'board-a', image_name: 'fast.png', imageCategory: 'general' }]);
      slow.resolve([{ boardId: 'board-a', image_name: 'slow.png', imageCategory: 'general' }]);
    });

    // Only the most recent click may win, regardless of resolution order.
    expect(mocks.selectItem.mock.calls.map((call) => call[0].image_name)).toEqual(['fast.png']);
  });

  it('ignores a click whose position lookup lands after a newer click', async () => {
    // The guard must hold across BOTH async hops: the hydrate and the
    // name-list fetch. A click whose names arrive late must not move the
    // gallery after a newer click has already landed it elsewhere.
    const slowNames = deferred<{ items: { kind: string; name: string }[]; starredCount: number; total: number }>();

    mocks.resolveMany
      .mockResolvedValueOnce([{ boardId: 'board-a', image_name: 'slow.png', imageCategory: 'general' }])
      .mockResolvedValueOnce([{ boardId: 'board-b', image_name: 'fast.png', imageCategory: 'general' }]);
    mocks.fetchQuery.mockReturnValueOnce(slowNames.promise).mockResolvedValueOnce(namesWithImageAt('fast.png', 0));
    await mount();

    await flush(() => handle.click?.('slow.png'));
    await flush(() => handle.click?.('fast.png'));
    await flush(() => slowNames.resolve(namesWithImageAt('slow.png', 0)));

    expect(mocks.selectBoard.mock.calls).toEqual([['board-b']]);
    expect(mocks.selectItem.mock.calls.map((call) => call[0].image_name)).toEqual(['fast.png']);
  });

  it('ignores a click left in flight across an unmount/remount', async () => {
    // The regression this guards: a per-mount counter is reset by the remount,
    // so the abandoned click compares against a dead counter, passes, and
    // overwrites the newer mount's selection. Reachable by switching the right
    // panel away from the map and back while a hydrate is in flight.
    const stale = deferred<{ boardId: string; image_name: string; imageCategory: string }[]>();
    const fresh = deferred<{ boardId: string; image_name: string; imageCategory: string }[]>();

    mocks.resolveMany.mockReturnValueOnce(stale.promise).mockReturnValueOnce(fresh.promise);

    await mount();
    await flush(() => handle.click?.('stale.png'));
    await unmount();

    await mount();
    await flush(() => handle.click?.('fresh.png'));

    await flush(() => {
      fresh.resolve([{ boardId: 'board-a', image_name: 'fresh.png', imageCategory: 'general' }]);
      stale.resolve([{ boardId: 'board-a', image_name: 'stale.png', imageCategory: 'general' }]);
    });

    expect(mocks.selectItem.mock.calls.map((call) => call[0].image_name)).toEqual(['fresh.png']);
  });

  it('leaves the selection alone when hydrate fails or the image is gone', async () => {
    mocks.resolveMany.mockRejectedValueOnce(new Error('deleted')).mockResolvedValueOnce([]);
    await mount();

    await flush(() => handle.click?.('gone.png'));
    await flush(() => handle.click?.('missing.png'));

    expect(mocks.selectItem).not.toHaveBeenCalled();
  });
});
