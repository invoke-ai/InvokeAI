/* oxlint-disable react-perf/jsx-no-new-object-as-prop */
import type { GalleryItemsFilter } from '@features/gallery/data/queries';

import { ChakraProvider } from '@chakra-ui/react';
import { EMPTY_GALLERY_MARKS_FILTER } from '@features/gallery/core/marks';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { GalleryUiAdapter } from './GalleryUiContext';
import type { GalleryWidgetContextValue } from './GalleryWidgetContext';

import { GalleryCullRow } from './GalleryCull';
import { GalleryUiProvider } from './GalleryUiContext';
import { GalleryWidgetContext } from './GalleryWidgetContext';

const backend = vi.hoisted(() => ({
  namesFilters: [] as GalleryItemsFilter[],
  rejected: [
    { kind: 'image' as const, name: 'blurry.png' },
    { kind: 'video' as const, name: 'shaky.mp4' },
  ],
  totalFilters: [] as GalleryItemsFilter[],
}));

vi.mock('react-i18next', () => ({
  useTranslation: () => ({ t: (key: string, values?: { count?: number }) => `${key}:${values?.count ?? ''}` }),
}));

vi.mock('@features/gallery/data/queries', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  galleryItemNamesOptions: (filter: GalleryItemsFilter) => ({
    queryFn: () => {
      backend.namesFilters.push(filter);

      return { items: backend.rejected, total: backend.rejected.length };
    },
    queryKey: ['test-names', filter],
  }),
  galleryItemsTotalOptions: (filter: GalleryItemsFilter) => ({
    queryFn: () => {
      backend.totalFilters.push(filter);

      return backend.rejected.length;
    },
    queryKey: ['test-total', filter],
  }),
}));

const itemActions = { deleteItems: vi.fn(async () => {}), setItemsMarks: vi.fn(async () => {}) };
let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const render = async (cullMode: boolean) => {
  const listingFilter: GalleryItemsFilter = {
    boardId: 'dogs',
    colorLabels: ['red'],
    flags: ['pick'],
    galleryView: 'images',
    minRating: 2,
    searchTerm: 'beach',
  };
  const widget = {
    filter: listingFilter,
    gallery: {
      marksFilter: { ...EMPTY_GALLERY_MARKS_FILTER, flags: ['pick'], rating: 2 },
      semanticImageQuery: null,
    },
    itemActions,
  } as unknown as GalleryWidgetContextValue;
  const adapter = {
    galleryValues: { cullMode },
    notifications: { add: vi.fn(), reportError: vi.fn() },
  } as unknown as GalleryUiAdapter;

  await act(() =>
    root?.render(
      <QueryClientProvider client={new QueryClient()}>
        <ChakraProvider value={system}>
          <GalleryUiProvider adapter={adapter}>
            <GalleryWidgetContext value={widget}>
              <GalleryCullRow />
            </GalleryWidgetContext>
          </GalleryUiProvider>
        </ChakraProvider>
      </QueryClientProvider>
    )
  );
};

beforeEach(() => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  backend.namesFilters.length = 0;
  backend.totalFilters.length = 0;
  itemActions.deleteItems.mockClear();
});

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('Delete rejected', () => {
  it('offers nothing outside Cull mode', async () => {
    await render(false);

    expect(host?.querySelector('[data-gallery-delete-rejected]')).toBeNull();
    expect(backend.totalFilters).toEqual([]);
  });

  it('sends every rejected item of the scope, whatever the marks filters, through the confirming deletion', async () => {
    await render(true);
    const button = await vi.waitFor(() => {
      const found = host?.querySelector<HTMLButtonElement>('[data-gallery-delete-rejected]');

      expect(found?.textContent).toContain('widgets.gallery.cull.deleteRejected:2');
      return found!;
    });

    await act(async () => {
      button.click();
      await vi.waitFor(() => expect(itemActions.deleteItems).toHaveBeenCalledOnce());
    });

    expect(backend.namesFilters).toEqual([
      expect.objectContaining({ boardId: 'dogs', flags: ['reject'], galleryView: 'images', searchTerm: 'beach' }),
    ]);
    expect(backend.namesFilters[0]).toMatchObject({
      colorLabels: undefined,
      maxRating: undefined,
      minRating: undefined,
      starred: undefined,
    });
    // Off screen, so the flow confirms even a single item with routine confirmations turned off.
    expect(itemActions.deleteItems).toHaveBeenCalledWith(
      backend.rejected,
      expect.objectContaining({ targetsOffScreen: true })
    );
  });
});
