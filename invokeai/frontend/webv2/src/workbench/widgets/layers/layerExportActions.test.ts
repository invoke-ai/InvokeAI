import { getGallerySettings } from '@features/gallery/contracts';
import {
  canonicalizeGalleryItemsFilter,
  galleryBoardsOptions,
  galleryKeys,
  getGalleryListingBoardsQuery,
} from '@features/gallery/queries';
import { accountLifecycle, captureAccountScope } from '@platform/state/accountLifecycle';
import { ApiError } from '@platform/transport/http';
import { QueryClient, QueryObserver } from '@tanstack/react-query';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { copyBlobToClipboard, copyLayerToClipboard, saveLayerToBoard } from './layerExportActions';

describe('copyBlobToClipboard', () => {
  it('writes a ClipboardItem using the blob MIME type', async () => {
    const write = vi.fn(() => Promise.resolve());
    const created: Record<string, Blob>[] = [];
    class TestClipboardItem {
      presentationStyle: PresentationStyle = 'unspecified';
      types: string[] = [];
      constructor(items: Record<string, Blob>) {
        created.push(items);
        this.types = Object.keys(items);
      }
      getType(type: string): Promise<Blob> {
        return Promise.resolve(created.at(-1)?.[type] ?? new Blob());
      }
    }
    const blob = new Blob(['pixels'], { type: 'image/png' });

    await copyBlobToClipboard(blob, { ClipboardItemCtor: TestClipboardItem, clipboard: { write } });

    expect(created).toHaveLength(1);
    expect(created[0]?.['image/png']).toBe(blob);
    expect(write).toHaveBeenCalledWith([expect.any(TestClipboardItem)]);
  });

  it('fails when the clipboard API is unavailable', async () => {
    await expect(copyBlobToClipboard(new Blob(['pixels']))).rejects.toThrow('Clipboard image copy is unavailable');
  });
});

describe('copyLayerToClipboard', () => {
  it('bakes the layer with disabled content and writes it; a refused export reports its status', async () => {
    const write = vi.fn(() => Promise.resolve());
    class TestClipboardItem {
      presentationStyle: PresentationStyle = 'unspecified';
      types: string[] = [];
      constructor(items: Record<string, Blob>) {
        this.types = Object.keys(items);
      }
      getType(): Promise<Blob> {
        return Promise.resolve(new Blob());
      }
    }
    const blob = new Blob(['pixels'], { type: 'image/png' });
    const exportLayer = vi.fn((layerId: string) =>
      Promise.resolve(layerId === 'ok' ? { blob, status: 'ok' as const } : { status: 'empty' as const })
    );
    const deps = { ClipboardItemCtor: TestClipboardItem, clipboard: { write }, exportLayer };

    await expect(copyLayerToClipboard('ok', deps)).resolves.toBe('ok');
    expect(exportLayer).toHaveBeenCalledWith('ok', { includeDisabled: true });
    expect(write).toHaveBeenCalledTimes(1);

    await expect(copyLayerToClipboard('blank', deps)).resolves.toBe('empty');
    expect(write).toHaveBeenCalledTimes(1);
  });
});

interface ServerBoard {
  id: string;
  actions: string[];
  archived?: boolean;
  imageCount?: number;
}

/** The board endpoints as the server answers them; every other request answers an empty total. */
const serveBoards = (initial: ServerBoard[]) => {
  let boards = initial;
  const listings: URLSearchParams[] = [];
  vi.stubGlobal(
    'fetch',
    vi.fn((input: RequestInfo | URL) => {
      const url = new URL(input instanceof Request ? input.url : String(input), 'http://invoke.test');
      if (url.pathname !== '/api/v1/boards/') {
        return Promise.resolve(new Response(JSON.stringify({ total: 0 })));
      }
      listings.push(url.searchParams);
      const includeArchived = url.searchParams.get('include_archived') === 'true';
      const body = boards
        .filter((board) => includeArchived || !board.archived)
        .map((board) => ({
          allowed_actions: board.actions,
          archived: board.archived ?? false,
          asset_count: 0,
          board_id: board.id,
          board_name: board.id,
          created_at: '2026-10-01T00:00:00Z',
          general_access: 'restricted',
          image_count: board.imageCount ?? 0,
          my_role: board.actions.includes('add_media') ? 'owner' : 'viewer',
        }));
      return Promise.resolve(new Response(JSON.stringify(body)));
    })
  );

  return {
    /** Reads of the listing of every readable board, archived included. */
    destinationReads: () => listings.filter((params) => params.get('include_archived') === 'true').length,
    listingReads: () => listings.length,
    setBoards: (next: ServerBoard[]) => {
      boards = next;
    },
  };
};

const createClient = () => new QueryClient({ defaultOptions: { queries: { gcTime: Infinity, retry: false } } });

const exportsPixels = () =>
  vi.fn(() => Promise.resolve({ blob: new Blob(['pixels'], { type: 'image/png' }), status: 'ok' as const }));

/** An item list the Gallery panel read for `boardId`, fresh for its 60-second stale time. */
const seedItemList = (client: QueryClient, boardId: string) => {
  const key = galleryKeys.items(
    captureAccountScope(),
    canonicalizeGalleryItemsFilter({ boardId, galleryView: 'images', searchTerm: '' })
  );
  client.setQueryData(key, { pageParams: [0], pages: [{ items: [], total: 0 }] });
  return () => client.getQueryState(key);
};

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('saveLayerToBoard', () => {
  it('uploads to a board the listing takes uploads for, then reads the gallery and its board counts again', async () => {
    const owner = accountLifecycle.activate('layer-export-a', ':user:layer-export-a');
    const server = serveBoards([
      { actions: ['read', 'add_media'], id: 'board-7' },
      { actions: ['read', 'add_media'], archived: true, id: 'old' },
    ]);
    const client = createClient();
    // A warm Gallery: the listing of every readable board, the panel's own listing (observed) and an item list.
    await client.fetchQuery(galleryBoardsOptions({ includeArchived: true }));
    const panel = new QueryObserver(client, galleryBoardsOptions(getGalleryListingBoardsQuery(getGallerySettings({}))));
    const unsubscribe = panel.subscribe(() => undefined);
    await vi.waitFor(() => expect(panel.getCurrentResult().data?.some((board) => board.id === 'board-7')).toBe(true));
    const itemList = seedItemList(client, 'board-7');
    const exportLayer = exportsPixels();
    let uploadedFile: File | null = null;
    const upload = vi.fn((file: File) => {
      uploadedFile = file;
      server.setBoards([
        { actions: ['read', 'add_media'], id: 'board-7', imageCount: 1 },
        { actions: ['read', 'add_media'], archived: true, id: 'old' },
      ]);
      return Promise.resolve();
    });

    await expect(saveLayerToBoard('layer-a', 'board-7', { client, exportLayer, upload })).resolves.toBe('saved');

    expect(exportLayer).toHaveBeenCalledWith('layer-a', { includeDisabled: true, signal: owner.signal });
    expect(upload).toHaveBeenCalledWith(expect.any(File), 'board-7', { signal: owner.signal });
    expect(uploadedFile).toMatchObject({ name: 'layer-layer-a.png', type: 'image/png' });
    await vi.waitFor(() =>
      expect(panel.getCurrentResult().data?.find((board) => board.id === 'board-7')?.imageCount).toBe(1)
    );
    expect(itemList()?.isInvalidated).toBe(true);
    unsubscribe();

    // An archived board chosen on purpose is a destination too.
    await expect(saveLayerToBoard('layer-a', 'old', { client, exportLayer, upload })).resolves.toBe('saved');
    expect(upload).toHaveBeenLastCalledWith(expect.any(File), 'old', { signal: owner.signal });
  });

  it('reads the listing of every readable board before deciding when none is cached yet', async () => {
    accountLifecycle.activate('layer-export-a', ':user:layer-export-a');
    const server = serveBoards([{ actions: ['read', 'add_media'], id: 'board-7' }]);
    const client = createClient();
    const readsWhenExported: number[] = [];
    const exportLayer = vi.fn(() => {
      readsWhenExported.push(server.destinationReads());
      return Promise.resolve({ blob: new Blob(['pixels'], { type: 'image/png' }), status: 'ok' as const });
    });
    const upload = vi.fn(() => Promise.resolve());

    await expect(saveLayerToBoard('layer-a', 'board-7', { client, exportLayer, upload })).resolves.toBe('saved');

    expect(readsWhenExported).toEqual([1]);
    expect(server.destinationReads()).toBe(1);
    expect(upload).toHaveBeenCalledWith(expect.any(File), 'board-7', { signal: expect.any(AbortSignal) });
  });

  it('reports a board revoked after it was listed, refreshes the gallery and stops offering the board', async () => {
    accountLifecycle.activate('layer-export-a', ':user:layer-export-a');
    const server = serveBoards([{ actions: ['read', 'add_media'], id: 'shared' }]);
    const client = createClient();
    await client.fetchQuery(galleryBoardsOptions({ includeArchived: true }));
    const itemList = seedItemList(client, 'shared');
    const exportLayer = exportsPixels();
    // Access dropped to view-only after the listing was read: the server refuses the upload.
    server.setBoards([{ actions: ['read'], id: 'shared' }]);
    const refusal = new ApiError('You cannot add to this board', 403);
    const upload = vi.fn(() => Promise.reject(refusal));

    await expect(saveLayerToBoard('layer-a', 'shared', { client, exportLayer, upload })).rejects.toBe(refusal);

    await vi.waitFor(() => expect(server.destinationReads()).toBe(2));
    await vi.waitFor(() => expect(client.isFetching()).toBe(0));
    expect(itemList()?.isInvalidated).toBe(true);

    // The next save is judged from the listing read again: refused before anything is baked or sent.
    await expect(saveLayerToBoard('layer-a', 'shared', { client, exportLayer, upload })).resolves.toBe(
      'board-view-only'
    );
    expect(exportLayer).toHaveBeenCalledTimes(1);
    expect(upload).toHaveBeenCalledTimes(1);
  });

  it('refuses a board the listing holds view-only or no longer holds, without baking, and reads the listing again', async () => {
    accountLifecycle.activate('layer-export-a', ':user:layer-export-a');
    const server = serveBoards([
      { actions: ['read', 'add_media'], id: 'own' },
      { actions: ['read'], id: 'viewed' },
    ]);
    const client = createClient();
    await client.fetchQuery(galleryBoardsOptions({ includeArchived: true }));
    const exportLayer = exportsPixels();
    const upload = vi.fn(() => Promise.resolve());

    await expect(saveLayerToBoard('layer-a', 'viewed', { client, exportLayer, upload })).resolves.toBe(
      'board-view-only'
    );
    await vi.waitFor(() => expect(server.destinationReads()).toBe(2));
    await vi.waitFor(() => expect(client.isFetching()).toBe(0));

    await expect(saveLayerToBoard('layer-a', 'deleted', { client, exportLayer, upload })).resolves.toBe(
      'board-unavailable'
    );
    await vi.waitFor(() => expect(server.destinationReads()).toBe(3));

    expect(exportLayer).not.toHaveBeenCalled();
    expect(upload).not.toHaveBeenCalled();
  });

  it('saves to Uncategorized without consulting any board listing', async () => {
    accountLifecycle.activate('layer-export-a', ':user:layer-export-a');
    const server = serveBoards([]);
    const client = createClient();
    const upload = vi.fn(() => Promise.resolve());

    await expect(saveLayerToBoard('layer-a', 'none', { client, exportLayer: exportsPixels(), upload })).resolves.toBe(
      'saved'
    );

    expect(upload).toHaveBeenCalledWith(expect.any(File), 'none', { signal: expect.any(AbortSignal) });
    expect(server.listingReads()).toBe(0);
  });

  it('does not upload an account A export that completes after account B activates', async () => {
    const owner = accountLifecycle.activate('layer-export-a', ':user:layer-export-a');
    let resolveExport!: (value: { status: 'empty' }) => void;
    const exportResult = new Promise<{ status: 'empty' }>((resolve) => {
      resolveExport = resolve;
    });
    const exportLayer = vi.fn(() => exportResult);
    const upload = vi.fn(() => Promise.resolve());
    const pendingSave = saveLayerToBoard('layer-a', 'none', { client: createClient(), exportLayer, upload });

    accountLifecycle.activate('layer-export-b', ':user:layer-export-b');
    resolveExport({ status: 'empty' });

    await expect(pendingSave).resolves.toBe('stale');
    expect(owner.signal.aborted).toBe(true);
    expect(upload).not.toHaveBeenCalled();
  });

  it('quietly drops an account A upload that completes after account B activates', async () => {
    const owner = accountLifecycle.activate('layer-export-a', ':user:layer-export-a');
    const exportLayer = vi.fn(() => Promise.resolve({ blob: new Blob(['pixels']), status: 'ok' as const }));
    let resolveUpload!: () => void;
    const uploadResult = new Promise<void>((resolve) => {
      resolveUpload = resolve;
    });
    const upload = vi.fn(() => uploadResult);
    const pendingSave = saveLayerToBoard('layer-a', 'none', { client: createClient(), exportLayer, upload });
    await vi.waitFor(() => expect(upload).toHaveBeenCalledOnce());

    accountLifecycle.activate('layer-export-b', ':user:layer-export-b');
    resolveUpload();

    await expect(pendingSave).resolves.toBe('stale');
    expect(owner.signal.aborted).toBe(true);
  });
});
