import type { GalleryItem, GalleryItemsPage } from '@features/gallery/core/items';

import { accountLifecycle } from '@platform/state/accountLifecycle';
import { QueryClient, QueryObserver } from '@tanstack/react-query';
import { afterEach, expect, it, vi } from 'vitest';

import { galleryItemsPageOptions } from './queries';
import { invalidateGallery } from './queryCache';
import { createGalleryRealtimeRuntime } from './realtimeRuntime';

const transport = vi.hoisted(() => ({ listItems: vi.fn() }));
vi.mock('./backend', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  listGalleryItems: transport.listItems,
}));

const cleanup: Array<() => void> = [];
afterEach(() => {
  for (const dispose of cleanup.splice(0).reverse()) {
    dispose();
  }
});

const delay = (ms: number) =>
  new Promise<void>((resolve) => {
    setTimeout(resolve, ms);
  });
const item = (name: string, boardId: string): GalleryItem => ({
  boardId,
  category: 'general',
  createdAt: '2026-11-05T00:00:00Z',
  fullUrl: `/full/${name}`,
  height: 64,
  isIntermediate: false,
  kind: 'image',
  name,
  starred: false,
  thumbnailUrl: `/thumb/${name}`,
  width: 64,
});

it('refreshes two independent listings over browser completion events without starving slow reads or revealing', async () => {
  accountLifecycle.activate('realtime-browser');
  let savedName = 'before.png';
  let slow = false;
  const completed = new Map<string, number>();
  const canceled = vi.fn();
  transport.listItems.mockReset().mockImplementation(async ({ boardId, signal }) => {
    const name = savedName;
    signal.addEventListener('abort', canceled, { once: true });
    try {
      if (slow) {
        await delay(1300);
      }
      signal.throwIfAborted();
      completed.set(boardId, (completed.get(boardId) ?? 0) + 1);
      return { items: [item(name, boardId)], total: 1 };
    } finally {
      signal.removeEventListener('abort', canceled);
    }
  });

  const channelName = `gallery-completions-${crypto.randomUUID()}`;
  const sender = new BroadcastChannel(channelName);
  cleanup.push(() => sender.close());
  const sessions = ['tab-one', 'tab-two'].map((boardId) => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    cleanup.push(() => client.clear());
    const observer = new QueryObserver(
      client,
      galleryItemsPageOptions({ boardId, galleryView: 'images', searchTerm: '' }, 0)
    );
    cleanup.push(observer.subscribe(() => undefined));
    const receiver = new BroadcastChannel(channelName);
    cleanup.push(() => receiver.close());
    const reveal = vi.fn();
    const invalidate = vi.fn(() => invalidateGallery(client));
    const runtime = createGalleryRealtimeRuntime({
      backend: {
        on: (event, handler) => {
          const listener = ({ data }: MessageEvent) => {
            if (data.event === event) {
              handler(data.payload as never);
            }
          };
          receiver.addEventListener('message', listener);
          return () => receiver.removeEventListener('message', listener);
        },
        onConnectionChange: () => () => undefined,
      },
      getSessionUserId: () => 'alice',
      invalidate,
      isClientQueueItemOrigin: (origin) => origin?.startsWith('webv2:') === true,
      onRevealRequest: reveal,
    });
    runtime.start();
    cleanup.push(runtime.dispose);
    return { boardId, observer, reveal, invalidate };
  });
  await vi.waitFor(() => expect(sessions.every(({ observer }) => observer.getCurrentResult().isSuccess)).toBe(true));
  expect(transport.listItems).toHaveBeenCalledTimes(2);
  const completion = (user_id = 'alice') =>
    sender.postMessage({
      event: 'queue_item_status_changed',
      payload: { destination: 'gallery:reveal', item_id: 1, origin: 'webv2:another-tab', status: 'completed', user_id },
    });
  slow = true;
  savedName = 'new-output.png';
  for (let i = 0; i < 20; i += 1) {
    completion();
  }
  // Continue beyond the one-second coalescing interval while the 1.3-second reads are still running.
  for (let i = 0; i < 12; i += 1) {
    await delay(100);
    completion();
  }
  await vi.waitFor(
    () => {
      for (const { boardId, observer } of sessions) {
        expect(completed.get(boardId)).toBeGreaterThanOrEqual(3);
        const data: GalleryItemsPage | undefined = observer.getCurrentResult().data;
        expect(data?.items[0]?.name).toBe('new-output.png');
        expect(observer.getCurrentResult().fetchStatus).toBe('idle');
      }
    },
    { timeout: 6000 }
  );
  expect(canceled).not.toHaveBeenCalled();
  expect(transport.listItems).toHaveBeenCalledTimes(6);
  for (const { reveal, invalidate } of sessions) {
    expect(reveal).not.toHaveBeenCalled();
    expect(invalidate).toHaveBeenCalledTimes(2);
  }
  completion('bob');
  await delay(300);
  expect(transport.listItems).toHaveBeenCalledTimes(6);
});
