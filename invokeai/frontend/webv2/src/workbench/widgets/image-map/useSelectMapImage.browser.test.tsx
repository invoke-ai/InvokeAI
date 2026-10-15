import type * as GalleryContracts from '@features/gallery/contracts';

import { accountLifecycle } from '@platform/state/accountLifecycle';
import { act, useEffect } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const mocks = vi.hoisted(() => ({
  activeProjectId: 'project-1',
  fetchBoards: vi.fn(),
  fetchVerifiedPage: vi.fn(),
  galleryValues: {} as Record<string, unknown>,
  patchValues: vi.fn(),
  registerImageCluster: vi.fn(),
  requestReveal: vi.fn(),
  resolve: vi.fn(),
  selectBoard: vi.fn(),
  selectItem: vi.fn(),
  locatorControllers: [] as AbortController[],
  setPage: vi.fn(),
  settings: { imageOrderDir: 'DESC', paginationMode: 'paginated' } as Record<string, unknown>,
  setView: vi.fn(),
}));

vi.mock('@features/gallery', () => ({
  galleryItems: { resolve: mocks.resolve },
  toGalleryItemKey: (ref: { kind: string; name: string }) => `${ref.kind}:${ref.name}`,
}));

vi.mock('@features/gallery/contracts', async (importOriginal) => ({
  // The navigation sequence is real: the staleness guarantees below are exactly
  // what it implements, and a stubbed counter would assert nothing.
  ...(await importOriginal<Record<string, unknown>>()),
  getGallerySettings: () => mocks.settings,
  registerImageCluster: mocks.registerImageCluster,
  requestGalleryItemReveal: mocks.requestReveal,
}));

vi.mock('@features/gallery/utility', () => ({
  abortGalleryLocatorRequests: () => {
    mocks.locatorControllers.forEach((controller) => controller.abort());
  },
  createGalleryLocatorRequest: () => {
    const controller = new AbortController();
    mocks.locatorControllers.push(controller);

    return { signal: controller.signal, release: vi.fn() };
  },
}));

vi.mock('@features/gallery/queries', () => ({
  GALLERY_PAGE_SIZE: 60,
  galleryBoardsOptions: (query: unknown) => ({ kind: 'boards', query, queryKey: ['boards', query] }),
  fetchVerifiedGalleryItemPage: (...args: unknown[]) => mocks.fetchVerifiedPage(...args),
}));

vi.mock('@tanstack/react-query', () => ({
  useQueryClient: () => ({
    fetchQuery: (options: { kind: string }) => mocks.fetchBoards(options),
  }),
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
  useWorkbenchQueries: () => ({
    getSnapshot: () => ({ activeProject: { id: mocks.activeProjectId } }),
    isActiveProject: (projectId: string) => projectId === mocks.activeProjectId,
  }),
}));

import { useClearClusterSelection, useMapSelection } from './useSelectMapImage';

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let host: HTMLDivElement | null = null;
let root: Root | null = null;

/** Published from an effect, not during render, so the probe stays render-pure. */
const handle: {
  click: ((item: { kind: 'image' | 'video'; name: string }) => void) | null;
  clickCluster:
    | ((
        primaryItem: { kind: 'image' | 'video'; name: string },
        itemKeys: `image:${string}`[] | `video:${string}`[] | (`image:${string}` | `video:${string}`)[],
        label: string
      ) => void)
    | null;
  clear: (() => void) | null;
} = { clear: null, click: null, clickCluster: null };

const Probe = () => {
  const { selectCluster, selectItem } = useMapSelection();
  const clearClusterSelection = useClearClusterSelection();

  useEffect(() => {
    handle.clear = clearClusterSelection;
    handle.click = selectItem;
    handle.clickCluster = selectCluster;
  }, [clearClusterSelection, selectCluster, selectItem]);

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

const verifiedAt = (index: number, total = index + 1) => ({
  index,
  offset: Math.floor(index / 60) * 60,
  page: { items: [], total },
  total,
});

beforeEach(() => {
  mocks.activeProjectId = 'project-1';
  mocks.galleryValues = {};
  mocks.settings = { imageOrderDir: 'DESC', paginationMode: 'paginated' };
  mocks.locatorControllers.length = 0;
  // Empty boards read as "still loading" — the reveal gives the board the
  // benefit of the doubt, matching the gallery's own fallback rules.
  mocks.fetchBoards.mockResolvedValue([]);
  mocks.fetchVerifiedPage.mockResolvedValue(verifiedAt(0));
  mocks.registerImageCluster.mockReturnValue('cluster-key-1');
});

afterEach(async () => {
  if (root) {
    await unmount();
  }
  mocks.fetchBoards.mockReset();
  mocks.fetchVerifiedPage.mockReset();
  mocks.patchValues.mockReset();
  mocks.registerImageCluster.mockReset();
  mocks.requestReveal.mockReset();
  mocks.resolve.mockReset();
  mocks.selectBoard.mockReset();
  mocks.selectItem.mockReset();
  mocks.setPage.mockReset();
  mocks.setView.mockReset();
});

describe('useMapSelection', () => {
  describe('selectItem', () => {
    it('dispatches the selection and a reveal for a click', async () => {
      mocks.resolve.mockResolvedValue({ boardId: 'board-a', category: 'general', kind: 'image', name: 'a.png' });
      await mount();

      await flush(() => handle.click?.({ kind: 'image', name: 'a.png' }));

      expect(mocks.selectItem).toHaveBeenCalledTimes(1);
      expect(mocks.selectItem.mock.calls[0]?.[0]).toEqual({
        boardId: 'board-a',
        category: 'general',
        kind: 'image',
        name: 'a.png',
      });
      // The reveal channel is what scrolls the grid; the selection alone must
      // not (auto-selected generation results would yank the scroll).
      expect(mocks.requestReveal).toHaveBeenCalledWith('image:a.png', expect.any(AbortSignal), 0);
    });

    it('reveals a clicked video through its own namespace', async () => {
      // Resolve videos through their own endpoint and item keys so gallery reveal matches.
      mocks.resolve.mockResolvedValue({ boardId: 'board-a', category: 'general', kind: 'video', name: 'clip.mp4' });
      mocks.fetchVerifiedPage.mockResolvedValue(verifiedAt(1, 2));
      await mount();

      await flush(() => handle.click?.({ kind: 'video', name: 'clip.mp4' }));

      expect(mocks.resolve).toHaveBeenCalledWith({ kind: 'video', name: 'clip.mp4' });
      expect(mocks.selectItem.mock.calls[0]?.[0]).toEqual({
        boardId: 'board-a',
        category: 'general',
        kind: 'video',
        name: 'clip.mp4',
      });
      expect(mocks.requestReveal).toHaveBeenCalledWith('video:clip.mp4', expect.any(AbortSignal), 1);
    });

    it('finds a video at its own position in a mixed listing', async () => {
      // The position lookup matches on kind as well as name. The two entries
      // are two pages apart, so matching on name alone lands the gallery on
      // the image's page and the clip is nowhere on screen.
      mocks.settings = { imageOrderDir: 'DESC', paginationMode: 'paginated' };
      mocks.resolve.mockResolvedValue({ boardId: 'board-a', category: 'general', kind: 'video', name: 'shared' });
      mocks.fetchVerifiedPage.mockResolvedValue(verifiedAt(120, 121));
      await mount();

      await flush(() => handle.click?.({ kind: 'video', name: 'shared' }));

      // Index 120 of a 60-per-page listing is page 2; the image's index 0 is page 0.
      expect(mocks.setPage).toHaveBeenCalledWith(2);
      expect(mocks.requestReveal).toHaveBeenCalledWith('video:shared', expect.any(AbortSignal), 120);
    });

    it("selects the image's board before the image itself", async () => {
      // Select the destination board before stamping navigation state so cross-board Preview paging has a cursor.
      mocks.resolve.mockResolvedValue({
        boardId: 'board-portraits',
        category: 'general',
        kind: 'image',
        name: 'a.png',
      });
      await mount();

      await flush(() => handle.click?.({ kind: 'image', name: 'a.png' }));

      expect(mocks.selectBoard).toHaveBeenCalledWith('board-portraits');
      expect(mocks.selectBoard.mock.invocationCallOrder[0]).toBeLessThan(mocks.selectItem.mock.invocationCallOrder[0]);
    });

    it('lands the gallery on the page holding the image in paginated mode', async () => {
      mocks.settings = { imageOrderDir: 'DESC', paginationMode: 'paginated' };
      mocks.resolve.mockResolvedValue({ boardId: 'board-a', category: 'general', kind: 'image', name: 'deep.png' });
      mocks.fetchVerifiedPage.mockResolvedValue(verifiedAt(130, 200));
      await mount();

      await flush(() => handle.click?.({ kind: 'image', name: 'deep.png' }));

      expect(mocks.setPage).toHaveBeenCalledWith(2);
      expect(mocks.selectItem).toHaveBeenCalledTimes(1);
      // The selection page rides along so the stamped navigation query
      // describes the page the image is actually on.
      expect(mocks.selectItem.mock.calls[0]?.[2]).toBe(2);
    });

    it("resolves the image's position against the listing the reveal lands on", async () => {
      mocks.settings = { imageOrderDir: 'ASC', paginationMode: 'paginated' };
      mocks.resolve.mockResolvedValue({ boardId: 'board-a', category: 'general', kind: 'image', name: 'a.png' });
      mocks.fetchVerifiedPage.mockResolvedValue(verifiedAt(0));
      await mount();

      await flush(() => handle.click?.({ kind: 'image', name: 'a.png' }));

      expect(mocks.fetchVerifiedPage.mock.calls[0]?.[1]).toEqual({
        boardId: 'board-a',
        galleryView: 'images',
        orderDir: 'ASC',
        searchTerm: '',
        starred: false,
      });
    });

    it('requests the resolved sparse slot directly in infinite mode', async () => {
      mocks.settings = { imageOrderDir: 'DESC', paginationMode: 'infinite' };
      mocks.resolve.mockResolvedValue({ boardId: 'board-a', category: 'general', kind: 'image', name: 'deep.png' });
      mocks.fetchVerifiedPage.mockResolvedValue(verifiedAt(130, 200));
      await mount();

      await flush(() => handle.click?.({ kind: 'image', name: 'deep.png' }));

      expect(mocks.setPage).not.toHaveBeenCalled();
      expect(mocks.requestReveal).toHaveBeenCalledWith('image:deep.png', expect.any(AbortSignal), 130);
      expect(mocks.selectItem.mock.calls[0]?.[2]).toBe(2);
    });

    it('does not require earlier pages to reveal a loaded deep target', async () => {
      mocks.settings = { imageOrderDir: 'DESC', paginationMode: 'infinite' };
      mocks.resolve.mockResolvedValue({ boardId: 'board-a', category: 'general', kind: 'image', name: 'deep.png' });
      mocks.fetchVerifiedPage.mockResolvedValue(verifiedAt(130, 200));
      await mount();

      await flush(() => handle.click?.({ kind: 'image', name: 'deep.png' }));

      expect(mocks.fetchVerifiedPage).toHaveBeenCalledOnce();
      expect(mocks.selectItem).toHaveBeenCalledTimes(1);
    });

    it('requests an absolute slot beyond the former infinite-window cap', async () => {
      mocks.settings = { imageOrderDir: 'DESC', paginationMode: 'infinite' };
      mocks.resolve.mockResolvedValue({ boardId: 'board-a', category: 'general', kind: 'image', name: 'deep.png' });
      mocks.fetchVerifiedPage.mockResolvedValue(verifiedAt(700, 800));
      await mount();

      await flush(() => handle.click?.({ kind: 'image', name: 'deep.png' }));

      expect(mocks.setPage).not.toHaveBeenCalled();
      expect(mocks.selectItem).toHaveBeenCalledTimes(1);
      expect(mocks.selectItem.mock.calls[0]?.[2]).toBe(11);
      expect(mocks.requestReveal).toHaveBeenCalledWith('image:deep.png', expect.any(AbortSignal), 700);
    });

    it('drops the page landing when the ordering settings changed mid-lookup', async () => {
      // The verified position describes the ordering it was fetched under;
      // changing sort before it settles must leave the Gallery untouched.
      const location = deferred<ReturnType<typeof verifiedAt>>();

      mocks.settings = { imageOrderDir: 'DESC', paginationMode: 'paginated' };
      mocks.resolve.mockResolvedValue({ boardId: 'board-a', category: 'general', kind: 'image', name: 'deep.png' });
      mocks.fetchVerifiedPage.mockReturnValue(location.promise);
      await mount();

      await flush(() => handle.click?.({ kind: 'image', name: 'deep.png' }));
      mocks.settings = { imageOrderDir: 'ASC', paginationMode: 'paginated' };
      await flush(() => location.resolve(verifiedAt(130, 200)));

      expect(mocks.setPage).not.toHaveBeenCalled();
      expect(mocks.selectBoard).not.toHaveBeenCalled();
      expect(mocks.selectItem).not.toHaveBeenCalled();
      expect(mocks.requestReveal).not.toHaveBeenCalled();
    });

    it('drops the page landing when the board is not listable in the gallery', async () => {
      // Do not apply hidden-board page positions to the Uncategorized fallback.
      mocks.settings = { imageOrderDir: 'DESC', paginationMode: 'paginated' };
      mocks.fetchBoards.mockResolvedValue([{ id: 'board-other' }]);
      mocks.resolve.mockResolvedValue({
        boardId: 'board-archived',
        category: 'general',
        kind: 'image',
        name: 'deep.png',
      });
      mocks.fetchVerifiedPage.mockResolvedValue(verifiedAt(130, 200));
      await mount();

      await flush(() => handle.click?.({ kind: 'image', name: 'deep.png' }));

      expect(mocks.setPage).not.toHaveBeenCalled();
      expect(mocks.selectBoard).toHaveBeenCalledWith('board-archived');
      expect(mocks.selectItem).toHaveBeenCalledTimes(1);
    });

    it('keeps the page landing when the boards lookup fails', async () => {
      mocks.settings = { imageOrderDir: 'DESC', paginationMode: 'paginated' };
      mocks.fetchBoards.mockRejectedValue(new Error('boards endpoint down'));
      mocks.resolve.mockResolvedValue({ boardId: 'board-a', category: 'general', kind: 'image', name: 'deep.png' });
      mocks.fetchVerifiedPage.mockResolvedValue(verifiedAt(130, 200));
      await mount();

      await flush(() => handle.click?.({ kind: 'image', name: 'deep.png' }));

      expect(mocks.setPage).toHaveBeenCalledWith(2);
    });

    it('clears an active search and similarity filter before revealing', async () => {
      // The image was located in the plain board listing; an active filter
      // would show some other list entirely.
      mocks.galleryValues = { searchTerm: 'sunset', semanticImageQuery: { kind: 'text', query: 'sunset' } };
      mocks.resolve.mockResolvedValue({ boardId: 'board-a', category: 'general', kind: 'image', name: 'a.png' });
      await mount();

      await flush(() => handle.click?.({ kind: 'image', name: 'a.png' }));

      expect(mocks.patchValues).toHaveBeenCalledWith('gallery', {
        searchTerm: '',
        semanticImageQuery: null,
        semanticSearchText: null,
        starredOnly: false,
      });
    });

    it('reveals a starred image in the starred listing it belongs to', async () => {
      mocks.resolve.mockResolvedValue({
        boardId: 'board-a',
        category: 'general',
        kind: 'image',
        name: 'a.png',
        starred: true,
      });
      mocks.fetchVerifiedPage.mockResolvedValue(verifiedAt(0));
      await mount();

      await flush(() => handle.click?.({ kind: 'image', name: 'a.png' }));

      expect(mocks.fetchVerifiedPage.mock.calls[0]?.[1]).toMatchObject({ boardId: 'board-a', starred: true });
      expect(mocks.patchValues).toHaveBeenCalledWith('gallery', {
        searchTerm: '',
        semanticImageQuery: null,
        semanticSearchText: null,
        starredOnly: true,
      });
    });

    it('leaves a semantic field, even an empty one, before revealing', async () => {
      // Semantic mode is a listing of its own even with nothing typed yet:
      // the reveal targets the board listing, so the field returns to it.
      mocks.galleryValues = { searchTerm: '', semanticImageQuery: null, semanticSearchText: '' };
      mocks.resolve.mockResolvedValue({ boardId: 'board-a', category: 'general', kind: 'image', name: 'a.png' });
      await mount();

      await flush(() => handle.click?.({ kind: 'image', name: 'a.png' }));

      expect(mocks.patchValues).toHaveBeenCalledWith('gallery', {
        searchTerm: '',
        semanticImageQuery: null,
        semanticSearchText: null,
        starredOnly: false,
      });
    });

    it('leaves the filters alone when none are active', async () => {
      mocks.resolve.mockResolvedValue({ boardId: 'board-a', category: 'general', kind: 'image', name: 'a.png' });
      await mount();

      await flush(() => handle.click?.({ kind: 'image', name: 'a.png' }));

      expect(mocks.patchValues).not.toHaveBeenCalled();
    });

    it('switches the gallery to the assets tab for a non-general image', async () => {
      mocks.resolve.mockResolvedValue({ boardId: 'board-a', category: 'user', kind: 'image', name: 'a.png' });
      await mount();

      await flush(() => handle.click?.({ kind: 'image', name: 'a.png' }));

      expect(mocks.setView).toHaveBeenCalledWith('assets');
    });

    it('does not touch the view when it already matches', async () => {
      mocks.resolve.mockResolvedValue({ boardId: 'board-a', category: 'general', kind: 'image', name: 'a.png' });
      await mount();

      await flush(() => handle.click?.({ kind: 'image', name: 'a.png' }));

      expect(mocks.setView).not.toHaveBeenCalled();
    });

    it('leaves Gallery state untouched when a verified position is unavailable', async () => {
      mocks.resolve.mockResolvedValue({ boardId: 'board-a', category: 'general', kind: 'image', name: 'a.png' });
      mocks.fetchVerifiedPage.mockResolvedValue(null);
      await mount();

      await flush(() => handle.click?.({ kind: 'image', name: 'a.png' }));

      expect(mocks.setPage).not.toHaveBeenCalled();
      expect(mocks.selectBoard).not.toHaveBeenCalled();
      expect(mocks.selectItem).not.toHaveBeenCalled();
      expect(mocks.requestReveal).not.toHaveBeenCalled();
    });

    it('does not touch the board for a click that never resolves an image', async () => {
      mocks.resolve.mockRejectedValue(new Error('not found'));
      await mount();

      await flush(() => handle.click?.({ kind: 'image', name: 'gone.png' }));

      expect(mocks.selectBoard).not.toHaveBeenCalled();
      expect(mocks.selectItem).not.toHaveBeenCalled();
      expect(mocks.requestReveal).not.toHaveBeenCalled();
    });
  });

  describe('selectCluster', () => {
    it('registers a cluster around a clicked video and reveals the clip', async () => {
      // Cluster mode has its own resolve and its own reveal key; a clip clicked
      // here must not be hydrated or revealed as an image.
      mocks.resolve.mockResolvedValue({ boardId: 'board-a', category: 'general', kind: 'video', name: 'clip.mp4' });
      await mount();

      await flush(() =>
        handle.clickCluster?.({ kind: 'video', name: 'clip.mp4' }, ['video:clip.mp4', 'image:a.png'], 'beaches')
      );

      expect(mocks.resolve).toHaveBeenCalledWith({ kind: 'video', name: 'clip.mp4' });
      expect(mocks.registerImageCluster).toHaveBeenCalledWith(['video:clip.mp4', 'image:a.png'], 'beaches');
      expect(mocks.requestReveal).toHaveBeenCalledWith('video:clip.mp4', expect.any(AbortSignal));
    });

    it('shows the cluster as a gallery filter with the clicked image selected and revealed', async () => {
      mocks.resolve.mockResolvedValue({ boardId: 'board-a', category: 'general', kind: 'image', name: 'a.png' });
      await mount();

      await flush(() =>
        handle.clickCluster?.(
          { kind: 'image', name: 'a.png' },
          ['image:a.png', 'image:b.png', 'image:c.png'],
          'beaches'
        )
      );

      expect(mocks.registerImageCluster).toHaveBeenCalledWith(['image:a.png', 'image:b.png', 'image:c.png'], 'beaches');
      expect(mocks.patchValues).toHaveBeenCalledWith('gallery', {
        galleryPage: 0,
        searchTerm: '',
        semanticImageQuery: { clusterId: 'cluster-key-1', kind: 'cluster', label: 'beaches' },
        semanticSearchText: null,
      });
      expect(mocks.selectItem).toHaveBeenCalledTimes(1);
      expect(mocks.selectItem.mock.calls[0]?.[0]).toEqual({
        boardId: 'board-a',
        category: 'general',
        kind: 'image',
        name: 'a.png',
      });
      // Re-clicking the same cluster point after scrolling away must return
      // the grid to the top; the reveal channel carries that even when the
      // selection is unchanged.
      expect(mocks.requestReveal).toHaveBeenCalledWith('image:a.png', expect.any(AbortSignal));
    });

    it("selects the primary image's board before the cluster filter", async () => {
      mocks.resolve.mockResolvedValue({
        boardId: 'board-landscapes',
        category: 'general',
        kind: 'image',
        name: 'a.png',
      });
      await mount();

      await flush(() =>
        handle.clickCluster?.({ kind: 'image', name: 'a.png' }, ['image:a.png', 'image:b.png'], 'label')
      );

      expect(mocks.selectBoard).toHaveBeenCalledWith('board-landscapes');
      expect(mocks.selectBoard.mock.invocationCallOrder[0]).toBeLessThan(mocks.selectItem.mock.invocationCallOrder[0]);
    });

    it('leaves the gallery alone when the primary image cannot be resolved', async () => {
      mocks.resolve.mockRejectedValue(new Error('not found'));
      await mount();

      await flush(() =>
        handle.clickCluster?.({ kind: 'image', name: 'gone.png' }, ['image:gone.png', 'image:b.png'], 'label')
      );

      expect(mocks.registerImageCluster).not.toHaveBeenCalled();
      expect(mocks.patchValues).not.toHaveBeenCalled();
      expect(mocks.selectItem).not.toHaveBeenCalled();
      expect(mocks.requestReveal).not.toHaveBeenCalled();
    });

    it('retires a cluster click still hydrating when its account epoch changes', async () => {
      accountLifecycle.activate('gallery-cluster-test');
      const pending = deferred<{ boardId: string; category: string; kind: string; name: string }>();
      mocks.resolve.mockReturnValueOnce(pending.promise);
      await mount();

      await flush(() => handle.clickCluster?.({ kind: 'image', name: 'a.png' }, ['image:a.png'], 'nearby'));
      accountLifecycle.activate('gallery-cluster-next-account');
      await flush(() => pending.resolve({ boardId: 'board-a', category: 'general', kind: 'image', name: 'a.png' }));

      expect(mocks.registerImageCluster).not.toHaveBeenCalled();
      expect(mocks.patchValues).not.toHaveBeenCalled();
      expect(mocks.selectBoard).not.toHaveBeenCalled();
      expect(mocks.selectItem).not.toHaveBeenCalled();
      expect(mocks.requestReveal).not.toHaveBeenCalled();
    });
  });

  it('shares one sequence guard across both modes, so the newer click wins', async () => {
    // The two entry points must not race each other: switching cluster mode
    // mid-flight would otherwise let a stale resolution overwrite a newer
    // selection.
    const slow = deferred<{ boardId: string; category: string; kind: string; name: string }>();
    const fast = deferred<{ boardId: string; category: string; kind: string; name: string }>();

    mocks.resolve.mockReturnValueOnce(slow.promise).mockReturnValueOnce(fast.promise);
    await mount();

    await flush(() => {
      handle.clickCluster?.({ kind: 'image', name: 'slow.png' }, ['image:slow.png'], 'label');
      handle.click?.({ kind: 'image', name: 'fast.png' });
    });
    await flush(() => {
      fast.resolve({ boardId: 'board-a', category: 'general', kind: 'image', name: 'fast.png' });
      slow.resolve({ boardId: 'board-a', category: 'general', kind: 'image', name: 'slow.png' });
    });

    expect(mocks.patchValues).not.toHaveBeenCalled();
    expect(mocks.selectItem).toHaveBeenCalledTimes(1);
    expect(mocks.selectItem.mock.calls[0]?.[0].name).toBe('fast.png');
  });

  it('retires a cluster click still hydrating when the selection is cleared', async () => {
    // A cluster listing is showing; the user clicks another cluster and clears
    // before that click's image resolves. The late resolution must not bring
    // the cleared selection back.
    // Through the real registry (the module's export is stubbed above): a
    // reference to an unregistered cluster reads as no cluster at all.
    const { registerImageCluster } = await vi.importActual<typeof GalleryContracts>('@features/gallery/contracts');
    const clusterId = registerImageCluster(['image:a.png', 'image:b.png'], 'beaches');
    mocks.galleryValues = { semanticImageQuery: { clusterId, kind: 'cluster', label: 'beaches' } };
    const slow = deferred<{ boardId: string; category: string; kind: string; name: string }>();

    mocks.resolve.mockReturnValueOnce(slow.promise);
    await mount();

    await flush(() => {
      handle.clickCluster?.({ kind: 'image', name: 'slow.png' }, ['image:slow.png'], 'forests');
      handle.clear?.();
    });
    await flush(() => {
      slow.resolve({ boardId: 'board-a', category: 'general', kind: 'image', name: 'slow.png' });
    });

    expect(mocks.patchValues).toHaveBeenCalledTimes(1);
    expect(mocks.patchValues).toHaveBeenCalledWith('gallery', expect.objectContaining({ semanticImageQuery: null }));
    expect(mocks.registerImageCluster).not.toHaveBeenCalled();
    expect(mocks.selectItem).not.toHaveBeenCalled();
  });

  it('leaves the gallery alone when a clear finds no cluster listing', async () => {
    mocks.galleryValues = { searchTerm: 'cats', semanticImageQuery: { imageName: 'a.png', kind: 'image' } };
    await mount();

    await flush(() => {
      handle.clear?.();
    });

    expect(mocks.patchValues).not.toHaveBeenCalled();
  });

  it('ignores a slow click that resolves after a newer one', async () => {
    const slow = deferred<{ boardId: string; category: string; kind: string; name: string }>();
    const fast = deferred<{ boardId: string; category: string; kind: string; name: string }>();

    mocks.resolve.mockReturnValueOnce(slow.promise).mockReturnValueOnce(fast.promise);
    await mount();

    await flush(() => {
      handle.click?.({ kind: 'image', name: 'slow.png' });
      handle.click?.({ kind: 'image', name: 'fast.png' });
    });

    await flush(() => {
      fast.resolve({ boardId: 'board-a', category: 'general', kind: 'image', name: 'fast.png' });
      slow.resolve({ boardId: 'board-a', category: 'general', kind: 'image', name: 'slow.png' });
    });

    // Only the most recent click may win, regardless of resolution order.
    expect(mocks.selectItem.mock.calls.map((call) => call[0].name)).toEqual(['fast.png']);
  });

  it('ignores a click whose verified page lands after a newer click', async () => {
    const slowLocation = deferred<ReturnType<typeof verifiedAt>>();

    mocks.resolve
      .mockResolvedValueOnce({ boardId: 'board-a', category: 'general', kind: 'image', name: 'slow.png' })
      .mockResolvedValueOnce({ boardId: 'board-b', category: 'general', kind: 'image', name: 'fast.png' });
    mocks.fetchVerifiedPage.mockReturnValueOnce(slowLocation.promise).mockResolvedValueOnce(verifiedAt(0));
    await mount();

    await flush(() => handle.click?.({ kind: 'image', name: 'slow.png' }));
    const slowSignal = mocks.fetchVerifiedPage.mock.calls[0]?.[4] as AbortSignal;
    await flush(() => handle.click?.({ kind: 'image', name: 'fast.png' }));
    expect(slowSignal.aborted).toBe(true);
    await flush(() => slowLocation.resolve(verifiedAt(1, 2)));

    expect(mocks.selectBoard.mock.calls).toEqual([['board-b']]);
    expect(mocks.selectItem.mock.calls.map((call) => call[0].name)).toEqual(['fast.png']);
  });

  it('ignores a click left in flight across an unmount/remount', async () => {
    // Fence selections across remounts; abandoned hydrations must not overwrite newer mount intents.
    const stale = deferred<{ boardId: string; category: string; kind: string; name: string }>();
    const fresh = deferred<{ boardId: string; category: string; kind: string; name: string }>();

    mocks.resolve.mockReturnValueOnce(stale.promise).mockReturnValueOnce(fresh.promise);

    await mount();
    await flush(() => handle.click?.({ kind: 'image', name: 'stale.png' }));
    await unmount();

    await mount();
    await flush(() => handle.click?.({ kind: 'image', name: 'fresh.png' }));

    await flush(() => {
      fresh.resolve({ boardId: 'board-a', category: 'general', kind: 'image', name: 'fresh.png' });
      stale.resolve({ boardId: 'board-a', category: 'general', kind: 'image', name: 'stale.png' });
    });

    expect(mocks.selectItem.mock.calls.map((call) => call[0].name)).toEqual(['fresh.png']);
  });

  it('drops a reveal whose hydrate landed after the user switched projects', async () => {
    // Fence all reveal writes to the original project even without a newer click.
    const inFlight = deferred<{ boardId: string; category: string; kind: string; name: string }>();

    mocks.resolve.mockReturnValueOnce(inFlight.promise);
    await mount();

    await flush(() => handle.click?.({ kind: 'image', name: 'left-behind.png' }));
    mocks.activeProjectId = 'project-2';

    await flush(() =>
      inFlight.resolve({ boardId: 'board-a', category: 'general', kind: 'image', name: 'left-behind.png' })
    );

    expect(mocks.selectBoard).not.toHaveBeenCalled();
    expect(mocks.selectItem).not.toHaveBeenCalled();
    expect(mocks.patchValues).not.toHaveBeenCalled();
  });

  it('leaves the selection alone when hydrate fails or the item is gone', async () => {
    mocks.resolve.mockRejectedValueOnce(new Error('deleted')).mockRejectedValueOnce(new Error('missing'));
    await mount();

    await flush(() => handle.click?.({ kind: 'image', name: 'gone.png' }));
    await flush(() => handle.click?.({ kind: 'image', name: 'missing.png' }));

    expect(mocks.selectItem).not.toHaveBeenCalled();
  });
});
