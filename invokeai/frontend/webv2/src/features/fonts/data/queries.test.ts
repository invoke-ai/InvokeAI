import type { FontCatalogPage, FontListParams, FontRecord } from '@features/fonts';

import { InfiniteQueryObserver, QueryClient } from '@tanstack/react-query';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const transport = vi.hoisted(() => ({
  listFonts: vi.fn(),
}));

vi.mock('./api', () => transport);

import { fontsInfiniteQueryOptions, fontsQueryOptions } from './queries';

const createFont = (id: string): FontRecord => ({
  axes: [],
  byteSize: 1024,
  contentHash: id,
  family: `Example ${id}`,
  filename: `${id}.ttf`,
  id,
  instances: [],
  label: `Example ${id} Regular`,
  scope: 'private',
  source: 'uploaded',
  style: 'normal',
  url: `/api/v1/fonts/${id}/file`,
  weight: 400,
});

describe('font catalog queries', () => {
  beforeEach(() => {
    transport.listFonts.mockReset();
  });

  it('fetches offset pages until the catalog total is covered', async () => {
    const firstPage: FontCatalogPage = {
      items: [createFont('first')],
      limit: 100,
      offset: 0,
      total: 101,
    };
    const secondPage: FontCatalogPage = {
      items: [createFont('second')],
      limit: 100,
      offset: 100,
      total: 101,
    };
    transport.listFonts.mockImplementation(({ offset }: FontListParams) =>
      Promise.resolve(offset === 0 ? firstPage : secondPage)
    );

    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const options = fontsInfiniteQueryOptions({ limit: 100, scope: 'all' });
    const observer = new InfiniteQueryObserver(queryClient, options);

    try {
      await observer.refetch();
      expect(observer.getCurrentResult().hasNextPage).toBe(true);
      await observer.fetchNextPage();

      const result = observer.getCurrentResult();
      expect(result.data?.pageParams).toEqual([0, 100]);
      expect(result.data?.pages.flatMap((page) => page.items.map((font) => font.id))).toEqual(['first', 'second']);
      expect(result.hasNextPage).toBe(false);
      expect(transport.listFonts.mock.calls.map(([params]) => params.offset)).toEqual([0, 100]);
    } finally {
      observer.destroy();
    }
  });

  it('keeps an offset-started catalog window on its own cache key', () => {
    const base = fontsInfiniteQueryOptions({ limit: 100, scope: 'all' });
    const offset = fontsInfiniteQueryOptions({ limit: 100, offset: 100, scope: 'all' });

    expect(offset.queryKey).not.toEqual(base.queryKey);
  });

  it('keeps normal and infinite catalog data isolated for matching filters', async () => {
    const page: FontCatalogPage = {
      items: [createFont('shared')],
      limit: 100,
      offset: 0,
      total: 1,
    };
    transport.listFonts.mockResolvedValue(page);

    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const params = { limit: 100, scope: 'all' } as const;
    const normal = fontsQueryOptions(params);
    const infinite = fontsInfiniteQueryOptions(params);

    await queryClient.fetchQuery(normal);
    await queryClient.fetchInfiniteQuery(infinite);

    expect(normal.queryKey).not.toEqual(infinite.queryKey);
    expect(queryClient.getQueryData(normal.queryKey)).toEqual(page);
    expect(queryClient.getQueryData(infinite.queryKey)).toEqual({ pages: [page], pageParams: [0] });
  });
});
