import type { GalleryItem } from '@features/gallery/core/items';

import { galleryStarredStripOptions } from '@features/gallery/data/queries';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { useGalleryStarredStrip } from './useGalleryStarredStrip';

vi.mock('@features/gallery/data/backend', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  // Never resolves: the cases below run entirely from the seeded cache.
  listGalleryItems: vi.fn(
    () =>
      new Promise(() => {
        /* pending */
      })
  ),
}));

const filter = { boardId: 'board-1', galleryView: 'images' as const, searchTerm: '' };
const starred: GalleryItem = {
  boardId: 'board-1',
  category: 'general',
  createdAt: '2026-07-30T00:00:00.000Z',
  fullUrl: '/full/starred.png',
  height: 64,
  isIntermediate: false,
  kind: 'image',
  name: 'starred.png',
  starred: true,
  thumbnailUrl: '/thumb/starred.png',
  width: 64,
};

const Probe = ({ enabled }: { enabled: boolean }) => {
  const strip = useGalleryStarredStrip({ enabled, filter });

  return (
    <output data-testid="strip">
      {JSON.stringify({ names: strip.items.map((item) => item.name), total: strip.total })}
    </output>
  );
};

let host: HTMLDivElement | null = null;
let root: Root | null = null;
let queryClient: QueryClient | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const renderProbe = async (enabled: boolean) => {
  await act(() =>
    root?.render(
      <QueryClientProvider client={queryClient!}>
        <Probe enabled={enabled} />
      </QueryClientProvider>
    )
  );

  return JSON.parse(host?.querySelector('[data-testid="strip"]')?.textContent ?? '{}') as {
    names: string[];
    total: number;
  };
};

beforeEach(() => {
  accountLifecycle.activate('starred-strip-hook-test');
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  queryClient = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
  // A strip page already in the cache, as after a board visit.
  queryClient.setQueryData(galleryStarredStripOptions(filter).queryKey, { items: [starred], total: 4 });
});

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  queryClient?.clear();
  host = null;
  root = null;
  queryClient = null;
});

describe('useGalleryStarredStrip', () => {
  it('reports the cached strip while enabled and empty the moment it is disabled', async () => {
    expect(await renderProbe(true)).toEqual({ names: ['starred.png'], total: 4 });
    // A disabled query still holds its cached data; the section must vanish
    // regardless (starred-only listing, ranked search, anchored window).
    expect(await renderProbe(false)).toEqual({ names: [], total: 0 });
    expect(await renderProbe(true)).toEqual({ names: ['starred.png'], total: 4 });
  });
});
