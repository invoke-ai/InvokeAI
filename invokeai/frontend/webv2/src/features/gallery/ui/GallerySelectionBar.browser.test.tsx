/* oxlint-disable react-perf/jsx-no-new-object-as-prop */
import type { GalleryItem } from '@features/gallery/contracts';
import type { GalleryBoard } from '@features/gallery/core/types';

import { ChakraProvider } from '@chakra-ui/react';
import { DEFAULT_GALLERY_SETTINGS } from '@features/gallery/core/settings';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { GalleryStateView } from './galleryStateView';
import type { GalleryWidgetContextValue } from './GalleryWidgetContext';

import { GallerySelectionBar } from './GallerySelectionBar';
import { GalleryWidgetContext } from './GalleryWidgetContext';

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    i18n: { language: 'en' },
    t: (key: string, values?: Record<string, unknown>) => {
      if (key === 'widgets.gallery.selectionCount') {
        return `${String(values?.count)} selected`;
      }
      if (key === 'widgets.gallery.uncategorized') {
        return 'Uncategorized';
      }

      return key;
    },
  }),
}));

const itemActions = {
  deleteItems: vi.fn(async () => {}),
  downloadItem: vi.fn(async () => {}),
  downloadItems: vi.fn(async () => {}),
  moveItemsToBoard: vi.fn(async () => {}),
  openItemInNewTab: vi.fn(),
  openItemInPreview: vi.fn(),
  setItemsStarred: vi.fn(async () => {}),
};

const createBoard = (overrides: Partial<GalleryBoard> & Pick<GalleryBoard, 'id' | 'name'>): GalleryBoard => ({
  archived: false,
  assetCount: 0,
  imageCount: 0,
  kind: 'board',
  projectId: null,
  videoCount: 0,
  ...overrides,
});

const createItem = (name: string, starred: boolean): GalleryItem => ({
  boardId: 'dogs',
  category: 'general',
  createdAt: '2026-07-30T00:00:00.000Z',
  fullUrl: `/full/${name}`,
  height: 64,
  isIntermediate: false,
  kind: 'image',
  name,
  starred,
  thumbnailUrl: `/thumb/${name}`,
  width: 64,
});

const createGallery = (overrides: Partial<GalleryStateView> = {}): GalleryStateView =>
  ({
    boards: [
      createBoard({ id: 'dogs', name: 'dogs' }),
      createBoard({ id: 'cats', name: 'Cats' }),
      createBoard({ id: 'none', kind: 'uncategorized', name: '' }),
      createBoard({ archived: true, id: 'old', name: 'Old' }),
      createBoard({ id: 'by_date:2026-07-30', kind: 'date', name: '30 July' }),
    ],
    compareImageKey: null,
    currentItem: null,
    galleryView: 'images',
    isLoading: false,
    items: [createItem('a.png', false), createItem('b.png', true)],
    pendingPlaceholders: [],
    projectBoardId: null,
    searchTerm: '',
    selectedBoardId: 'dogs',
    selectedItemKey: 'image:a.png',
    selectedItemKeys: ['image:a.png', 'image:b.png'],
    settings: DEFAULT_GALLERY_SETTINGS,
    ...overrides,
  }) as GalleryStateView;

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const renderBar = async (gallery: GalleryStateView) => {
  const contextValue = { gallery, itemActions } as unknown as GalleryWidgetContextValue;

  await act(() =>
    root?.render(
      <ChakraProvider value={system}>
        <GalleryWidgetContext value={contextValue}>
          <GallerySelectionBar />
        </GalleryWidgetContext>
      </ChakraProvider>
    )
  );
};

const getButton = (label: string): HTMLButtonElement => {
  const button = document.querySelector<HTMLButtonElement>(`button[aria-label="${label}"]`);

  if (!button) {
    throw new Error(`no button labelled ${label}`);
  }

  return button;
};

const click = async (element: HTMLElement) => {
  await act(async () => {
    element.click();
    await Promise.resolve();
  });
};

beforeEach(() => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  Object.values(itemActions).forEach((mock) => mock.mockClear());
});

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('GallerySelectionBar', () => {
  it('stays out of the way until something is selected', async () => {
    await renderBar(createGallery({ selectedItemKeys: [] }));

    expect(document.querySelector('[role="toolbar"]')).toBeNull();
  });

  it('reports the selection size', async () => {
    await renderBar(createGallery());

    expect(host?.textContent).toContain('2 selected');
  });

  it('stars the whole selection when any member is unstarred', async () => {
    await renderBar(createGallery());
    await click(getButton('widgets.gallery.starSelection'));

    expect(itemActions.setItemsStarred).toHaveBeenCalledExactlyOnceWith(
      [
        { kind: 'image', name: 'a.png' },
        { kind: 'image', name: 'b.png' },
      ],
      true
    );
  });

  it('unstars only once every member is already starred', async () => {
    await renderBar(createGallery({ items: [createItem('a.png', true), createItem('b.png', true)] }));
    await click(getButton('widgets.gallery.unstarSelection'));

    expect(itemActions.setItemsStarred).toHaveBeenCalledWith(expect.anything(), false);
  });

  it('downloads and deletes the selection', async () => {
    const gallery = createGallery();

    await renderBar(gallery);
    await click(getButton('widgets.gallery.downloadSelection'));
    await click(getButton('widgets.gallery.deleteSelection'));

    expect(itemActions.downloadItems).toHaveBeenCalledWith(
      [
        { kind: 'image', name: 'a.png' },
        { kind: 'image', name: 'b.png' },
      ],
      gallery.items
    );
    expect(itemActions.deleteItems).toHaveBeenCalledWith([
      { kind: 'image', name: 'a.png' },
      { kind: 'image', name: 'b.png' },
    ]);
  });

  it('offers only boards worth moving to — never a date board, an archived board, or where the items already are', async () => {
    await renderBar(createGallery());
    await click(getButton('widgets.gallery.moveSelectionToBoard'));

    const labels = Array.from(document.querySelectorAll('[data-part="item-text"]')).map(
      (element) => element.textContent
    );

    expect(labels).toEqual(['Cats', 'Uncategorized']);
  });

  it('disables the move control when there is nowhere to move to', async () => {
    await renderBar(createGallery({ boards: [createBoard({ id: 'dogs', name: 'dogs' })] }));

    expect(getButton('widgets.gallery.moveSelectionToBoard').disabled).toBe(true);
  });
});
