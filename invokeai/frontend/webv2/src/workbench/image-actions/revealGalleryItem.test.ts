import type { GalleryItem } from '@features/gallery/core/items';
import type { QueryClient } from '@tanstack/react-query';
import type { WorkbenchCommands, WorkbenchQueries } from '@workbench/workbenchStore';

import { claimGalleryNavigationSequence } from '@features/gallery/contracts';
import { captureAccountScope, accountLifecycle } from '@platform/state/accountLifecycle';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { GalleryRevealTicket } from './revealGalleryItem';

const mocks = vi.hoisted(() => ({
  fetchVerifiedPage: vi.fn(),
  galleryValues: {} as Record<string, unknown>,
  resolve: vi.fn(),
  settings: {
    boardOrderBy: 'created_at',
    boardOrderDir: 'DESC',
    imageOrderDir: 'DESC',
    paginationMode: 'paginated',
    showArchivedBoards: false,
    showDateBoards: false,
  } as Record<string, unknown>,
  requestReveal: vi.fn(),
}));

vi.mock('@features/gallery', () => ({
  galleryItems: { resolve: mocks.resolve },
  toGalleryItemKey: (ref: { kind: string; name: string }) => `${ref.kind}:${ref.name}`,
}));

vi.mock('@features/gallery/contracts', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  getGallerySettings: () => mocks.settings,
  requestGalleryItemReveal: mocks.requestReveal,
}));

vi.mock('@features/gallery/queries', () => ({
  GALLERY_PAGE_SIZE: 60,
  fetchVerifiedGalleryItemPage: (...args: unknown[]) => mocks.fetchVerifiedPage(...args),
  galleryBoardsOptions: (query: unknown) => ({ kind: 'boards', query }),
}));

vi.mock('@workbench/widgetState', () => ({ getProjectWidgetValues: () => mocks.galleryValues }));

import { revealGalleryItem } from './revealGalleryItem';

const item: GalleryItem = {
  boardId: 'board-1',
  category: 'general',
  createdAt: '2026-07-15T12:00:00Z',
  fullUrl: '/images/target.png',
  height: 64,
  isIntermediate: false,
  kind: 'image',
  name: 'target.png',
  starred: false,
  thumbnailUrl: '/images/target.webp',
  width: 64,
};

const queryClient = {
  fetchQuery: vi.fn().mockResolvedValue([{ id: 'board-1' }]),
} as unknown as QueryClient;

const createContext = () => {
  const commands = {
    gallery: {
      selectBoard: vi.fn(),
      selectItem: vi.fn(),
      setPage: vi.fn(),
      setView: vi.fn(),
    },
    widgets: { patchValues: vi.fn() },
  } as unknown as WorkbenchCommands;
  const queries = {
    getSnapshot: () => ({ activeProject: { id: 'project-1' } }),
    isActiveProject: (projectId: string) => projectId === 'project-1',
  } as unknown as WorkbenchQueries;

  return { context: { commands, queryClient, queries }, commands };
};

const ticket = (): GalleryRevealTicket => ({
  accountScope: captureAccountScope(),
  locatorSignal: new AbortController().signal,
  projectId: 'project-1',
  sequence: claimGalleryNavigationSequence(),
});

beforeEach(() => {
  accountLifecycle.activate('gallery-reveal-test');
  mocks.fetchVerifiedPage
    .mockReset()
    .mockResolvedValue({ index: 127, offset: 120, page: { items: [item], total: 200 }, total: 200 });
  mocks.galleryValues = {};
  mocks.resolve.mockReset().mockResolvedValue(item);
  mocks.settings = {
    boardOrderBy: 'created_at',
    boardOrderDir: 'DESC',
    imageOrderDir: 'DESC',
    paginationMode: 'paginated',
    showArchivedBoards: false,
    showDateBoards: false,
  };
  mocks.requestReveal.mockReset();
  vi.mocked(queryClient.fetchQuery)
    .mockReset()
    .mockResolvedValue([{ id: 'board-1' }]);
});

afterEach(() => accountLifecycle.invalidate());

describe('revealGalleryItem', () => {
  it('verifies the exact ordinary listing page before changing Gallery state and sends the absolute slot to the grid', async () => {
    const { commands, context } = createContext();
    const revealTicket = ticket();

    await revealGalleryItem(context, { kind: 'image', name: item.name }, revealTicket);

    expect(mocks.fetchVerifiedPage).toHaveBeenCalledWith(
      queryClient,
      { boardId: 'board-1', galleryView: 'images', orderDir: 'DESC', searchTerm: '', starred: false },
      { kind: 'image', name: 'target.png' },
      expect.objectContaining({ accountId: 'gallery-reveal-test' }),
      revealTicket.locatorSignal
    );
    expect(commands.gallery.selectBoard).toHaveBeenCalledWith('board-1');
    expect(commands.gallery.setPage).toHaveBeenCalledWith(2);
    expect(commands.gallery.selectItem).toHaveBeenCalledWith(item, 'project-1', 2);
    expect(mocks.requestReveal).toHaveBeenCalledWith('image:target.png', revealTicket.accountScope.signal, 127);
  });

  it('keeps selection, filters, board, and scroll unchanged when the one locator retry cannot verify a page', async () => {
    mocks.fetchVerifiedPage.mockResolvedValue(null);
    mocks.galleryValues = { searchTerm: 'sunset', semanticImageQuery: { kind: 'text', query: 'sunset' } };
    const { commands, context } = createContext();

    await revealGalleryItem(context, { kind: 'image', name: item.name }, ticket());

    expect(commands.widgets.patchValues).not.toHaveBeenCalled();
    expect(commands.gallery.selectBoard).not.toHaveBeenCalled();
    expect(commands.gallery.setView).not.toHaveBeenCalled();
    expect(commands.gallery.setPage).not.toHaveBeenCalled();
    expect(commands.gallery.selectItem).not.toHaveBeenCalled();
    expect(mocks.requestReveal).not.toHaveBeenCalled();
  });

  it('resolves semantic or cluster selections against the ordinary destination listing before clearing the query', async () => {
    mocks.galleryValues = {
      searchTerm: '',
      semanticImageQuery: { clusterId: 'cluster-1', kind: 'cluster', label: 'nearby' },
      semanticSearchText: '',
    };
    const { commands, context } = createContext();

    await revealGalleryItem(context, { kind: 'image', name: item.name }, ticket());

    expect(mocks.fetchVerifiedPage.mock.calls[0]?.[1]).not.toHaveProperty('semanticQuery');
    expect(commands.widgets.patchValues).toHaveBeenCalledWith('gallery', {
      searchTerm: '',
      semanticImageQuery: null,
      semanticSearchText: null,
      starredOnly: false,
    });
    expect(commands.gallery.selectItem).toHaveBeenCalledOnce();
  });

  it('discards a gesture when its captured account epoch changes during media resolution', async () => {
    let resolveItem!: (resolved: GalleryItem) => void;
    mocks.resolve.mockReturnValue(
      new Promise((resolve) => {
        resolveItem = resolve;
      })
    );
    const { commands, context } = createContext();
    const reveal = revealGalleryItem(context, { kind: 'image', name: item.name }, ticket());

    accountLifecycle.activate('gallery-reveal-next-account');
    resolveItem(item);

    await expect(reveal).rejects.toThrow('account scope');
    expect(mocks.fetchVerifiedPage).not.toHaveBeenCalled();
    expect(commands.gallery.selectBoard).not.toHaveBeenCalled();
    expect(commands.gallery.selectItem).not.toHaveBeenCalled();
  });

  it('does not resolve media when the originating gesture account changed before the reveal chunk runs', async () => {
    const { commands, context } = createContext();
    const originatingTicket = ticket();

    accountLifecycle.activate('gallery-reveal-next-account');
    await revealGalleryItem(context, { kind: 'image', name: item.name }, originatingTicket);

    expect(mocks.resolve).not.toHaveBeenCalled();
    expect(mocks.fetchVerifiedPage).not.toHaveBeenCalled();
    expect(commands.gallery.selectBoard).not.toHaveBeenCalled();
    expect(commands.gallery.selectItem).not.toHaveBeenCalled();
  });
});
