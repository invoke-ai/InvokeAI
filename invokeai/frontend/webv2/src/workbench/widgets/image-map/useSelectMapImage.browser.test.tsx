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
  locatorReleases: [] as ReturnType<typeof vi.fn>[],
  setPage: vi.fn(),
  settings: { imageOrderDir: 'DESC', paginationMode: 'paginated' } as Record<string, unknown>,
  setView: vi.fn(),
}));

vi.mock('@features/gallery', () => ({
  galleryItems: { resolve: mocks.resolve },
  toGalleryItemKey: (ref: { kind: string; name: string }) => `${ref.kind}:${ref.name}`,
}));

vi.mock('@features/gallery/contracts', async (importOriginal) => ({
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
    const release = vi.fn();
    mocks.locatorControllers.push(controller);
    mocks.locatorReleases.push(release);

    return { signal: controller.signal, release };
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
  getResolvedProjectWidgetValues: () => mocks.galleryValues,
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
  mocks.locatorReleases.length = 0;
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
      expect(mocks.requestReveal).toHaveBeenCalledWith('image:a.png', expect.any(AbortSignal), 0);
    });

    it("selects the image's board before the image itself", async () => {
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
      expect(mocks.selectItem.mock.calls[0]?.[2]).toBe(2);
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

    it('drops the page landing when the ordering settings changed mid-lookup', async () => {
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

    it('clears an active search and similarity filter before revealing', async () => {
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

    it('switches the gallery to the assets tab for a non-general image', async () => {
      mocks.resolve.mockResolvedValue({ boardId: 'board-a', category: 'user', kind: 'image', name: 'a.png' });
      await mount();

      await flush(() => handle.click?.({ kind: 'image', name: 'a.png' }));

      expect(mocks.setView).toHaveBeenCalledWith('assets');
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
  });

  describe('selectCluster', () => {
    it('registers a cluster around a clicked video and reveals the clip', async () => {
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
      expect(mocks.requestReveal).toHaveBeenCalledWith('image:a.png', expect.any(AbortSignal));
    });

    it('holds a locator request while its primary image hydrates, so no unrequested reveal supersedes it', async () => {
      const pending = deferred<{ boardId: string; category: string; kind: string; name: string }>();
      mocks.resolve.mockReturnValueOnce(pending.promise);
      await mount();

      await flush(() => handle.clickCluster?.({ kind: 'image', name: 'a.png' }, ['image:a.png'], 'nearby'));

      expect(mocks.locatorReleases).toHaveLength(1);
      expect(mocks.locatorReleases[0]).not.toHaveBeenCalled();

      await flush(() => pending.resolve({ boardId: 'board-a', category: 'general', kind: 'image', name: 'a.png' }));

      expect(mocks.selectItem).toHaveBeenCalledOnce();
      expect(mocks.locatorReleases[0]).toHaveBeenCalledOnce();
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

  it('ignores a click left in flight across an unmount/remount', async () => {
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
