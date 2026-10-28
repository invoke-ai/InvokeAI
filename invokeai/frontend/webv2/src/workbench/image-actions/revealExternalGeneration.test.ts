import type { GalleryItemRef } from '@features/gallery/contracts';
import type { GalleryItem } from '@features/gallery/core/items';
import type { QueueItem } from '@features/queue/contracts';
import type { QueryClient } from '@tanstack/react-query';
import type { WorkbenchCommands, WorkbenchQueries } from '@workbench/workbenchStore';

import { claimGalleryNavigationSequence } from '@features/gallery/contracts';
import { getGalleryRevealRequest } from '@features/gallery/core/selection';
import { createGalleryLocatorRequest } from '@features/gallery/queries';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { ExternalGenerationRevealContext } from './revealExternalGeneration';

import { attachExternalGenerationReveal } from './recallParametersBridge';

const mocks = vi.hoisted(() => ({
  fetchVerifiedPage: vi.fn(),
  readLastSavedOutput: vi.fn<(itemId: number, signal?: AbortSignal) => Promise<GalleryItemRef | null>>(),
  resolve: vi.fn(),
}));

vi.mock('@features/queue/queries', () => ({ readQueueItemLastSavedOutput: mocks.readLastSavedOutput }));
vi.mock('@features/gallery', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  galleryItems: { resolve: mocks.resolve },
}));
vi.mock('@features/gallery/queries', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  fetchVerifiedGalleryItemPage: (...args: unknown[]) => mocks.fetchVerifiedPage(...args),
}));

const createItem = ({ kind, name }: GalleryItemRef): GalleryItem => {
  const base = {
    boardId: 'board-1',
    category: 'general',
    createdAt: '2026-10-15T12:00:00Z',
    fullUrl: `/${kind}s/${name}`,
    height: 64,
    isIntermediate: false,
    name,
    starred: false,
    thumbnailUrl: `/thumbs/${name}`,
    width: 64,
  } as const;

  return kind === 'video' ? { ...base, durationSeconds: 4, kind } : { ...base, kind };
};

const lastImage: GalleryItemRef = { kind: 'image', name: 'last.png' };

const verifiedPage = (ref: GalleryItemRef) =>
  Promise.resolve({ index: 0, offset: 0, page: { items: [createItem(ref)], total: 1 }, total: 1 });

/** A promise the test settles; lets a test act while the reveal waits on the backend. */
const deferred = <T>() => {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((settle) => {
    resolve = settle;
  });
  return { promise, resolve };
};

let activeProjectId = 'project-1';
let galleryValues: Record<string, unknown> = {};
let queueItems: Pick<QueueItem, 'status'>[] = [];
let visible = true;
let editingText = false;
const visibilityListeners = new Set<() => void>();
/** The browser's `visibilitychange`: the tab is shown or hidden. */
const setVisible = (next: boolean) => {
  visible = next;
  for (const listener of visibilityListeners) {
    listener();
  }
};
let commands: {
  gallery: Record<'selectBoard' | 'selectItem' | 'setPage' | 'setView', ReturnType<typeof vi.fn>>;
  widgets: { patchValues: ReturnType<typeof vi.fn> };
};

const createContext = (): ExternalGenerationRevealContext => ({
  commands: commands as unknown as WorkbenchCommands,
  isEditingText: () => editingText,
  isVisible: () => visible,
  onVisibilityChange: (listener) => {
    visibilityListeners.add(listener);
    return () => visibilityListeners.delete(listener);
  },
  queries: {
    getSnapshot: () => ({
      activeProject: {
        id: activeProjectId,
        queue: { items: queueItems },
        widgetInstances: { gallery: { state: { values: galleryValues }, typeId: 'gallery' } },
      },
    }),
    isActiveProject: (projectId: string) => projectId === activeProjectId,
  } as unknown as WorkbenchQueries,
  queryClient: { fetchQuery: vi.fn().mockResolvedValue([{ id: 'board-1' }]) } as unknown as QueryClient,
});

/**
 * Attach the composing bridge with the real reveal module (and through it the real Find in Gallery path), recording
 * each reveal run so a test can wait for all of them to settle.
 */
const attach = (loadGate: Promise<void> = Promise.resolve()) => {
  const runs: Promise<unknown>[] = [];
  const revealer = attachExternalGenerationReveal({
    context: createContext(),
    load: async () => {
      await loadGate;
      const module = await import('./revealExternalGeneration');
      return {
        revealExternalGeneration: (...args: Parameters<typeof module.revealExternalGeneration>) => {
          const run = module.revealExternalGeneration(...args);
          runs.push(run.catch(() => undefined));
          return run;
        },
      };
    },
  });
  const settle = async (count: number) => {
    await vi.waitFor(() => expect(runs).toHaveLength(count));
    await Promise.all(runs);
  };

  return { revealer, settle };
};

const reveal = async (queueItemId = 42) => {
  const { revealer, settle } = attach();

  revealer.request(queueItemId);
  await settle(1);
  revealer.dispose();
};

const revealed = () => commands.gallery.selectItem.mock.calls.map(([item]) => (item as GalleryItem).name);

beforeEach(() => {
  accountLifecycle.activate('external-reveal-test');
  activeProjectId = 'project-1';
  galleryValues = { selectedBoardId: 'board-0', selectedImageName: 'image:mine.png' };
  queueItems = [];
  visible = true;
  editingText = false;
  visibilityListeners.clear();
  commands = {
    gallery: { selectBoard: vi.fn(), selectItem: vi.fn(), setPage: vi.fn(), setView: vi.fn() },
    widgets: { patchValues: vi.fn() },
  };
  mocks.readLastSavedOutput.mockReset().mockResolvedValue(lastImage);
  mocks.resolve.mockReset().mockImplementation((ref: GalleryItemRef) => Promise.resolve(createItem(ref)));
  mocks.fetchVerifiedPage
    .mockReset()
    .mockImplementation((_client: unknown, _filter: unknown, ref: GalleryItemRef) => verifiedPage(ref));
});

afterEach(() => accountLifecycle.invalidate());

describe('external generation reveal', () => {
  it("lands the Gallery on the item's last saved image and its board", async () => {
    await reveal(42);

    expect(mocks.readLastSavedOutput).toHaveBeenCalledWith(42, expect.any(AbortSignal));
    expect(commands.gallery.selectBoard).toHaveBeenCalledWith('board-1');
    expect(revealed()).toEqual(['last.png']);
    expect(getGalleryRevealRequest()).toMatchObject({
      announcement: 'external-generation',
      itemKey: 'image:last.png',
    });
  });

  it('lands on a saved video, showing both kinds when the media filter hides videos', async () => {
    galleryValues = { ...galleryValues, mediaFilter: 'image' };
    mocks.readLastSavedOutput.mockResolvedValue({ kind: 'video', name: 'clip.mp4' });

    await reveal();

    expect(commands.widgets.patchValues).toHaveBeenCalledWith(
      'gallery',
      expect.objectContaining({ mediaFilter: 'all' })
    );
    expect(revealed()).toEqual(['clip.mp4']);
    expect(getGalleryRevealRequest()).toMatchObject({ itemKey: 'video:clip.mp4' });
  });

  it.each([
    ['the user is typing', () => (editingText = true)],
    ['a run of their own is open', () => (queueItems = [{ status: 'running' }])],
    [
      'the user has several items selected',
      () => (galleryValues = { ...galleryValues, selectedImageNames: ['image:a.png', 'image:b.png'] }),
    ],
    ['a text search narrows the listing', () => (galleryValues = { ...galleryValues, searchTerm: 'cat' })],
    ['a semantic query narrows the listing', () => (galleryValues = { ...galleryValues, semanticSearchText: 'cat' })],
    [
      'a similarity query narrows the listing',
      () => (galleryValues = { ...galleryValues, semanticImageQuery: { kind: 'text', query: 'cat' } }),
    ],
  ])('leaves the Gallery alone, reading no results, when %s', async (_, arrange) => {
    arrange();

    await reveal();

    expect(mocks.readLastSavedOutput).not.toHaveBeenCalled();
    expect(revealed()).toEqual([]);
  });

  it('never supersedes a Find in Gallery still locating its item', async () => {
    const find = createGalleryLocatorRequest();

    await reveal();

    expect(find.signal.aborted).toBe(false);
    expect(revealed()).toEqual([]);
    find.release();
  });

  it.each([
    ['selects another image', () => (galleryValues = { ...galleryValues, selectedImageName: 'image:other.png' })],
    ['switches boards', () => (galleryValues = { ...galleryValues, selectedBoardId: 'board-2' })],
    ['starts a Find in Gallery', () => claimGalleryNavigationSequence()],
    ['starts typing', () => (editingText = true)],
    ['starts a run of their own', () => (queueItems = [{ status: 'pending' }])],
    ['switches projects', () => (activeProjectId = 'project-2')],
  ])('makes no Gallery change when the user %s while the result is read', async (_, act) => {
    mocks.readLastSavedOutput.mockImplementation(() => {
      act();
      return Promise.resolve(lastImage);
    });

    await reveal();

    expect(commands.gallery.selectBoard).not.toHaveBeenCalled();
    expect(revealed()).toEqual([]);
  });

  it.each([
    ['selects another image', () => (galleryValues = { ...galleryValues, selectedImageName: 'image:other.png' })],
    ['starts a Find in Gallery', () => claimGalleryNavigationSequence()],
    ['starts typing', () => (editingText = true)],
    ['starts a run of their own', () => (queueItems = [{ status: 'pending' }])],
  ])('makes no Gallery change when the user %s while the item is located', async (_, act) => {
    mocks.fetchVerifiedPage.mockImplementation((_client: unknown, _filter: unknown, ref: GalleryItemRef) => {
      act();
      return verifiedPage(ref);
    });

    await reveal();

    expect(mocks.fetchVerifiedPage).toHaveBeenCalled();
    expect(commands.gallery.selectBoard).not.toHaveBeenCalled();
    expect(commands.widgets.patchValues).not.toHaveBeenCalled();
    expect(revealed()).toEqual([]);
  });

  it('does not reveal once the account that received the completion is gone', async () => {
    mocks.readLastSavedOutput.mockImplementation(() => {
      accountLifecycle.invalidate();
      return Promise.resolve(lastImage);
    });

    await reveal();

    expect(revealed()).toEqual([]);
  });

  it('reveals nothing for an item without a saved image, such as a workflow-call child', async () => {
    mocks.readLastSavedOutput.mockResolvedValue(null);

    await reveal();

    expect(revealed()).toEqual([]);
  });

  it('lets only the newest of a burst of completions reveal, even when an older read finishes last', async () => {
    const reads = new Map([
      [1, deferred<GalleryItemRef | null>()],
      [2, deferred<GalleryItemRef | null>()],
    ]);
    mocks.readLastSavedOutput.mockImplementation((itemId) => reads.get(itemId)!.promise);
    const { revealer, settle } = attach();

    revealer.request(1);
    await vi.waitFor(() => expect(mocks.readLastSavedOutput).toHaveBeenCalledTimes(1));
    revealer.request(2);
    await vi.waitFor(() => expect(mocks.readLastSavedOutput).toHaveBeenCalledTimes(2));
    reads.get(2)!.resolve({ kind: 'image', name: 'newest.png' });
    await vi.waitFor(() => expect(revealed()).toEqual(['newest.png']));
    reads.get(1)!.resolve({ kind: 'image', name: 'older.png' });
    await settle(2);

    expect(revealed()).toEqual(['newest.png']);
    revealer.dispose();
  });

  it('drops a superseded completion before reading its results', async () => {
    const { revealer, settle } = attach();

    revealer.request(1);
    revealer.request(2);
    await settle(2);

    expect(mocks.readLastSavedOutput.mock.calls.map(([itemId]) => itemId)).toEqual([2]);
    revealer.dispose();
  });

  it('drops a reveal still in flight when the runtime is disposed', async () => {
    const read = deferred<GalleryItemRef | null>();
    mocks.readLastSavedOutput.mockReturnValue(read.promise);
    const { revealer, settle } = attach();

    revealer.request(42);
    await vi.waitFor(() => expect(mocks.readLastSavedOutput).toHaveBeenCalled());
    revealer.dispose();
    read.resolve(lastImage);
    await settle(1);

    expect(mocks.readLastSavedOutput.mock.calls[0]![1]!.aborted).toBe(true);
    expect(revealed()).toEqual([]);
  });

  it('reveals through an open but empty semantic field, which Find in Gallery then closes', async () => {
    galleryValues = { ...galleryValues, semanticImageQuery: null, semanticSearchText: '' };

    await reveal();

    expect(revealed()).toEqual(['last.png']);
    expect(commands.widgets.patchValues).toHaveBeenCalledWith(
      'gallery',
      expect.objectContaining({ semanticSearchText: null })
    );
  });

  it('fences the reveal by the navigation at arrival, even while its chunk is still loading', async () => {
    const loading = deferred<void>();
    const { revealer, settle } = attach(loading.promise);

    revealer.request(42);
    galleryValues = { ...galleryValues, selectedImageName: 'image:other.png' };
    loading.resolve();
    await settle(1);

    expect(mocks.readLastSavedOutput).not.toHaveBeenCalled();
    expect(revealed()).toEqual([]);
    revealer.dispose();
  });

  it.each([
    ['galleryPage', 3],
    ['galleryView', 'assets'],
    ['mediaFilter', 'video'],
    ['searchTerm', ''],
    ['selectedBoardId', 'board-2'],
    ['selectedImageName', 'image:other.png'],
    ['selectedImageNames', ['image:other.png']],
    ['semanticSearchText', ''],
    ['starredOnly', true],
  ])('treats a change to %s after arrival as the user navigating', async (key, value) => {
    mocks.readLastSavedOutput.mockImplementation(() => {
      galleryValues = { ...galleryValues, [key]: value };
      return Promise.resolve(lastImage);
    });

    await reveal();

    expect(revealed()).toEqual([]);
  });

  describe('in a hidden tab', () => {
    it('waits until the tab is shown, then reveals', async () => {
      visible = false;
      const { revealer, settle } = attach();

      revealer.request(42);
      await settle(1);

      expect(mocks.readLastSavedOutput).not.toHaveBeenCalled();

      setVisible(true);
      await settle(2);

      expect(revealed()).toEqual(['last.png']);
      revealer.dispose();
    });

    it('resumes a reveal the tab hid while its item was being located', async () => {
      mocks.fetchVerifiedPage.mockImplementationOnce((_client: unknown, _filter: unknown, ref: GalleryItemRef) => {
        visible = false;
        return verifiedPage(ref);
      });
      const { revealer, settle } = attach();

      revealer.request(42);
      await settle(1);

      expect(revealed()).toEqual([]);

      setVisible(true);
      await settle(2);

      expect(revealed()).toEqual(['last.png']);
      revealer.dispose();
    });

    it('keeps only the newest request waiting', async () => {
      visible = false;
      const { revealer, settle } = attach();

      revealer.request(1);
      revealer.request(2);
      await settle(2);
      setVisible(true);
      await settle(3);

      expect(mocks.readLastSavedOutput.mock.calls.map(([itemId]) => itemId)).toEqual([2]);
      revealer.dispose();
    });

    it('drops a waiting request whose user navigated before the tab was shown', async () => {
      visible = false;
      const { revealer, settle } = attach();

      revealer.request(42);
      await settle(1);
      galleryValues = { ...galleryValues, selectedBoardId: 'board-2' };
      setVisible(true);
      await settle(2);

      expect(revealed()).toEqual([]);
      revealer.dispose();
    });

    it('stops listening for the tab, and drops the waiting request, on dispose', async () => {
      visible = false;
      const { revealer, settle } = attach();

      revealer.request(42);
      await settle(1);
      revealer.dispose();

      expect(visibilityListeners.size).toBe(0);
      expect(mocks.readLastSavedOutput).not.toHaveBeenCalled();
    });
  });
});
