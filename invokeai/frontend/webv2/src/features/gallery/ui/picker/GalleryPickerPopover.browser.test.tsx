/* oxlint-disable react-perf/jsx-no-new-object-as-prop, react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop */
import type { GalleryItem } from '@features/gallery/core/items';
import type { GalleryBoard } from '@features/gallery/core/types';
import type { GalleryUiAdapter } from '@features/gallery/react';
import type { GalleryHost } from '@features/gallery/ui/GalleryUiContext';

import { ChakraProvider } from '@chakra-ui/react';
import { getGalleryUploadAccept, legacyGeneratedImageToGalleryItem } from '@features/gallery/core/items';
import { GalleryUiProvider } from '@features/gallery/react';
import { GalleryHostProvider } from '@features/gallery/ui/GalleryUiContext';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { act, useCallback, useState } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { getGalleryPickerSelectionAfterPick, type GalleryPickerSelection } from './galleryPicker';
import { GalleryPickerPopover } from './GalleryPickerPopover';

const mocks = vi.hoisted(() => ({
  invalidateGallery: vi.fn(),
  listItems: vi.fn(),
  listInfiniteItems: vi.fn(),
  uploadGalleryImage: vi.fn(),
  uploadGalleryVideo: vi.fn(),
}));

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    i18n: { language: 'en' },
    t: (key: string, values?: Record<string, unknown>) =>
      values && 'name' in values ? `${key}:${String(values.name)}` : key,
  }),
}));

vi.mock('@features/queue/react', () => ({
  useQueueItemProgress: () => null,
  useQueueItemProgressImage: () => null,
}));

vi.mock('@features/gallery/data/queries', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  galleryBoardsOptions: () => ({
    queryFn: () => Promise.resolve(boards),
    queryKey: ['test-picker-boards'],
    staleTime: Infinity,
  }),
  galleryItemsInfiniteOptions: (filter: { boardId: string; galleryView: string; searchTerm: string }) => ({
    getNextPageParam: () => undefined,
    initialPageParam: 0,
    queryFn: () => Promise.resolve(mocks.listInfiniteItems(filter)),
    queryKey: ['test-picker-items', filter.boardId, filter.galleryView, filter.searchTerm],
    staleTime: Infinity,
  }),
  galleryItemsPageOptions: (filter: { boardId: string; galleryView: string; searchTerm: string }, offset: number) => ({
    queryFn: () => Promise.resolve(mocks.listItems({ ...filter, offset })),
    queryKey: ['gallery', 'items', 'list', { accountId: 'test-account', epoch: 0 }, filter, 'page', offset],
    staleTime: Infinity,
  }),
}));

vi.mock('@features/gallery/data/backend', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  uploadGalleryImage: (...args: unknown[]) => mocks.uploadGalleryImage(...args),
  uploadGalleryVideo: (...args: unknown[]) => mocks.uploadGalleryVideo(...args),
}));

vi.mock('@features/gallery/data/queryCache', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  invalidateGallery: (...args: unknown[]) => mocks.invalidateGallery(...args),
}));

const board: GalleryBoard = {
  archived: false,
  assetCount: 0,
  assetVideoCount: 0,
  id: 'dogs',
  imageCount: 3,
  kind: 'board',
  name: 'Dogs',
  isInbox: false,
  projectId: null,
  videoCount: 1,
};
const uncategorized: GalleryBoard = {
  ...board,
  assetCount: 1,
  id: 'none',
  imageCount: 1,
  kind: 'uncategorized',
  name: '',
};
/** Another project's board: the Gallery widget hides these by default, the picker lists them. */
const otherProjectBoard: GalleryBoard = { ...board, id: 'cats', imageCount: 0, name: 'Cats', projectId: 'project-b' };
const boards = [board, uncategorized, otherProjectBoard];

/** Newest first, matching the picker's DESC order. */
const ORDER = ['a.png', 'b.png', 'c.mp4', 'cat.png', 'e.png', 'f.png', 'loose.png'];

const image = (name: string, boardId = 'dogs'): GalleryItem => ({
  boardId,
  category: 'general',
  createdAt: `2026-09-0${9 - ORDER.indexOf(name)}T00:00:00.000Z`,
  fullUrl: `/full/${name}`,
  height: 64,
  isIntermediate: false,
  kind: 'image',
  name,
  starred: false,
  thumbnailUrl: `/thumb/${name}`,
  width: 64,
});
const video = (name: string): GalleryItem => ({ ...image(name), durationSeconds: 4, kind: 'video' });

const dogItems: GalleryItem[] = [
  image('a.png'),
  image('b.png'),
  video('c.mp4'),
  image('cat.png'),
  image('e.png'),
  image('f.png'),
];
const uncategorizedItems: GalleryItem[] = [image('loose.png', 'none')];

let host: HTMLDivElement | null = null;
let root: Root | null = null;
let queryClient: QueryClient | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const galleryCommands = { selectBoard: vi.fn(), selectItem: vi.fn(), setView: vi.fn() };
const openGallery = vi.fn(() => true);
const notifications = { add: vi.fn(), reportError: vi.fn() };

const adapter = {
  gallery: galleryCommands,
  galleryValues: { galleryView: 'images', selectedBoardId: 'dogs', selectedImage: dogItems[1] },
  notifications,
  projectName: 'Project',
  widgets: { openGallery, patchGalleryValues: vi.fn() },
} as unknown as GalleryUiAdapter;

const onPick = vi.fn();

/** A consumer that records picks into `selection`, as every multi-add slot does. */
const MultiAddHost = ({ initialSelection }: { initialSelection: GalleryPickerSelection }) => {
  const [selection, setSelection] = useState(initialSelection);
  const handlePick = useCallback((item: GalleryItem) => {
    onPick(item);
    setSelection((current) => getGalleryPickerSelectionAfterPick(current, item));
  }, []);

  return (
    <GalleryPickerPopover accept={['image']} label="Add images" selection={selection} onPick={handlePick}>
      <button type="button">Add images</button>
    </GalleryPickerPopover>
  );
};

const OPEN_DIALOG = '[role="dialog"][data-state="open"]';

const settle = async () => {
  for (let round = 0; round < 6; round += 1) {
    await act(async () => {
      await new Promise((resolve) => {
        requestAnimationFrame(() => setTimeout(resolve, 0));
      });
    });
  }
};

/** A host without a workbench (the Launchpad): no Gallery widget scope to seed from or reveal. */
const launchpadHost: GalleryHost = { galleryValues: {}, notifications, projectName: '' };

const renderPicker = async (picker?: React.ReactElement, { withoutWorkbench = false } = {}) => {
  const content = picker ?? (
    <GalleryPickerPopover accept={['image']} label="Choose image" onPick={onPick}>
      <button type="button">Choose image</button>
    </GalleryPickerPopover>
  );
  await act(() =>
    root?.render(
      <ChakraProvider value={system}>
        <QueryClientProvider client={queryClient!}>
          {withoutWorkbench ? (
            <GalleryHostProvider host={launchpadHost}>{content}</GalleryHostProvider>
          ) : (
            <GalleryUiProvider adapter={adapter}>{content}</GalleryUiProvider>
          )}
        </QueryClientProvider>
      </ChakraProvider>
    )
  );

  const trigger = host?.querySelector<HTMLButtonElement>('button');

  if (!trigger) {
    throw new Error('trigger did not render');
  }

  return trigger;
};

const openPicker = async (picker?: React.ReactElement, options?: { withoutWorkbench?: boolean }) => {
  const trigger = await renderPicker(picker, options);

  await act(() => trigger.click());
  await settle();

  const dialog = document.querySelector<HTMLElement>(OPEN_DIALOG);

  if (!dialog) {
    throw new Error('picker did not open');
  }

  // The view is lazy-loaded on first open.
  await vi.waitFor(() => expect(dialog.querySelector('input:not([type="file"])')).not.toBeNull(), { timeout: 10_000 });
  await settle();

  return { dialog, trigger };
};

const getSearchInput = (dialog: HTMLElement) => {
  const input = dialog.querySelector<HTMLInputElement>('input:not([type="file"])');

  if (!input) {
    throw new Error('search input missing');
  }

  return input;
};

const getOptions = (dialog: HTMLElement) => [...dialog.querySelectorAll<HTMLElement>('[role="option"]')];
const getOption = (dialog: HTMLElement, key: string) => {
  const option = dialog.querySelector<HTMLElement>(`[data-item-key="${CSS.escape(key)}"]`);

  if (!option) {
    throw new Error(`option ${key} missing`);
  }

  return option;
};
const getActiveOption = (dialog: HTMLElement) =>
  document.getElementById(getSearchInput(dialog).getAttribute('aria-activedescendant') ?? '');
const getStatus = (dialog: HTMLElement) => dialog.querySelector('[role="status"]')?.textContent ?? '';
const getBoardsRegion = (dialog: HTMLElement) =>
  dialog.querySelector<HTMLElement>('[role="region"][aria-label="widgets.gallery.picker.boardsLabel"]');
const getBoardRow = (dialog: HTMLElement, text: string) =>
  [...(getBoardsRegion(dialog)?.querySelectorAll<HTMLButtonElement>('button') ?? [])].find((row) =>
    row.textContent?.includes(text)
  );
const getColumnCount = (dialog: HTMLElement) => {
  const listbox = dialog.querySelector('[role="listbox"]');
  const firstVirtualRow = listbox?.firstElementChild;

  return getComputedStyle(firstVirtualRow ?? listbox!).gridTemplateColumns.split(' ').length;
};

const pressKey = async (target: HTMLElement, key: string) => {
  await act(() => target.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key })));
  await settle();
};

const typeSearch = async (input: HTMLInputElement, value: string) => {
  const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set;

  await act(() => {
    setter?.call(input, value);
    input.dispatchEvent(new Event('input', { bubbles: true }));
  });
  await settle();
};

beforeEach(() => {
  vi.clearAllMocks();
  mocks.listItems.mockImplementation((filter: { boardId: string; offset: number; searchTerm: string }) => {
    const items = filter.boardId === 'none' ? uncategorizedItems : dogItems;
    const matching = items.filter((item) => item.name.includes(filter.searchTerm));
    const pageItems = matching.slice(filter.offset, filter.offset + 60);

    return {
      itemIndices: pageItems.map((_, index) => filter.offset + index),
      items: pageItems,
      offset: filter.offset,
      total: matching.length,
    };
  });
  host = document.createElement('div');
  document.body.appendChild(host);
  root = createRoot(host);
  queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
});

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
  queryClient?.clear();
});

describe('GalleryPickerPopover', () => {
  it('opens with the search focused and the gallery selection ready to pick with Enter', async () => {
    const { dialog, trigger } = await openPicker();
    const input = getSearchInput(dialog);

    await vi.waitFor(() => expect(document.activeElement).toBe(input));
    expect(dialog.getAttribute('aria-label')).toBe('Choose image');
    expect(getOptions(dialog)).toHaveLength(6);
    expect(getActiveOption(dialog)).toBe(getOption(dialog, 'image:b.png'));
    expect(getOption(dialog, 'image:b.png').getAttribute('aria-selected')).toBe('true');
    expect(getStatus(dialog)).toBe('widgets.gallery.picker.itemCount');

    const activeTab = dialog.querySelector<HTMLElement>('[role="tab"][aria-selected="true"]');

    expect(document.getElementById(activeTab?.getAttribute('aria-controls') ?? '')?.getAttribute('role')).toBe(
      'tabpanel'
    );

    await pressKey(input, 'Enter');

    expect(onPick).toHaveBeenCalledExactlyOnceWith(dogItems[1]);
    expect(document.querySelector(OPEN_DIALOG)).toBeNull();
    await vi.waitFor(() => expect(document.activeElement).toBe(trigger));
  });

  it('moves the highlight with the arrow keys, by column for up and down, and picks with a click', async () => {
    const { dialog } = await openPicker();
    const input = getSearchInput(dialog);
    const columns = getColumnCount(dialog);

    expect(columns).toBeGreaterThanOrEqual(3);

    await pressKey(input, 'ArrowLeft');
    expect(getActiveOption(dialog)).toBe(getOption(dialog, 'image:a.png'));

    await pressKey(input, 'ArrowDown');
    expect(getActiveOption(dialog)?.dataset.itemKey).toBe(`image:${ORDER[columns]}`);

    await pressKey(input, 'ArrowUp');
    await pressKey(input, 'ArrowRight');
    await pressKey(input, 'ArrowRight');
    expect(getActiveOption(dialog)).toBe(getOption(dialog, 'video:c.mp4'));
    expect(getStatus(dialog)).toBe('widgets.gallery.picker.unsupportedVideo');

    await pressKey(input, 'Enter');
    expect(onPick).not.toHaveBeenCalled();

    await act(() => getOption(dialog, 'image:cat.png').click());
    expect(onPick).toHaveBeenCalledExactlyOnceWith(dogItems[3]);
    expect(document.querySelector(OPEN_DIALOG)).toBeNull();
  });

  it('keeps the highlight on its item when an insert shifts the listing', async () => {
    const { dialog } = await openPicker();

    expect(getActiveOption(dialog)).toBe(getOption(dialog, 'image:b.png'));

    // Another client adds a newer image, so the refetched listing moves every item one slot along.
    dogItems.unshift(image('new.png'));
    try {
      await act(() => queryClient!.invalidateQueries());
      await vi.waitFor(() => expect(getOption(dialog, 'image:new.png')).toBeDefined());
      await settle();

      expect(getActiveOption(dialog)).toBe(getOption(dialog, 'image:b.png'));

      await pressKey(getSearchInput(dialog), 'Enter');
      expect(onPick).toHaveBeenCalledExactlyOnceWith(expect.objectContaining({ name: 'b.png' }));
    } finally {
      dogItems.shift();
    }
  });

  it('requests a distant keyboard target by page offset without fetching intervening pages', async () => {
    const total = 2_400;
    const lastPageOffset = Math.floor((total - 1) / 60) * 60;

    mocks.listItems.mockImplementation((filter: { offset: number }) => {
      const items = Array.from({ length: Math.min(60, total - filter.offset) }, (_, index) =>
        image(`distant-${filter.offset + index}.png`)
      );

      return {
        itemIndices: items.map((_, index) => filter.offset + index),
        items,
        offset: filter.offset,
        total,
      };
    });

    const { dialog } = await openPicker();
    const input = getSearchInput(dialog);

    await pressKey(input, 'End');
    await vi.waitFor(() => expect(getActiveOption(dialog)?.getAttribute('aria-posinset')).toBe(String(total)));

    const requestedOffsets = mocks.listItems.mock.calls.map(([filter]) => filter.offset as number);

    expect(requestedOffsets).toContain(0);
    expect(requestedOffsets).toContain(lastPageOffset);
    expect(requestedOffsets.every((offset) => offset === 0 || offset >= lastPageOffset - 60)).toBe(true);
    expect(mocks.listInfiniteItems).not.toHaveBeenCalled();
  });

  it('holds the item a placeholder highlight lands on once its page loads, through a later insert', async () => {
    let inserted = 0;
    mocks.listItems.mockImplementation((filter: { offset: number }) => {
      const total = 2_390 + inserted;
      const items = Array.from({ length: Math.max(0, Math.min(60, total - filter.offset)) }, (_, index) => {
        const position = filter.offset + index - inserted;

        return image(position < 0 ? 'adopted-new.png' : `adopted-${position}.png`);
      });

      return { itemIndices: items.map((_, index) => filter.offset + index), items, offset: filter.offset, total };
    });

    const { dialog } = await openPicker();
    const input = getSearchInput(dialog);

    // End lands on a placeholder: its page is requested only now.
    await pressKey(input, 'End');
    await vi.waitFor(() => expect(getActiveOption(dialog)?.dataset.itemKey).toBe('image:adopted-2389.png'));

    inserted = 1;
    await act(() => queryClient!.invalidateQueries());
    await vi.waitFor(() => expect(dialog.querySelector('[role="option"][aria-setsize="2391"]')).not.toBeNull());
    await settle();

    expect(getActiveOption(dialog)?.dataset.itemKey).toBe('image:adopted-2389.png');
    await pressKey(input, 'Enter');
    expect(onPick).toHaveBeenCalledExactlyOnceWith(expect.objectContaining({ name: 'adopted-2389.png' }));
  });

  it('keeps both pages of a view that straddles a page edge loaded while the arrows move within it', async () => {
    const total = 120;
    mocks.listItems.mockImplementation((filter: { offset: number }) => {
      const items = Array.from({ length: Math.max(0, Math.min(60, total - filter.offset)) }, (_, index) =>
        image(`edge-${filter.offset + index}.png`)
      );

      return { itemIndices: items.map((_, index) => filter.offset + index), items, offset: filter.offset, total };
    });

    const { dialog } = await openPicker();
    const input = getSearchInput(dialog);
    const activeIndex = () => Number(getActiveOption(dialog)?.dataset.itemIndex ?? -1);
    const columns = getColumnCount(dialog);

    await pressKey(input, 'End');
    await vi.waitFor(() => expect(activeIndex()).toBe(total - 1));

    // Walk up into the first page: the row just left behind, on the second page, stays in view.
    while (activeIndex() >= 60) {
      await pressKey(input, 'ArrowUp');
    }
    const belowIndex = activeIndex() + columns;

    expect(belowIndex).toBeGreaterThanOrEqual(60);
    // Record every state the tile passes through: dropping its page would flash a skeleton before it re-subscribes.
    const belowKey = `image:edge-${belowIndex}.png`;
    const tileKeys = new Set<string | null | undefined>();
    const observer = new MutationObserver(() => {
      tileKeys.add(dialog.querySelector(`[data-item-index="${belowIndex}"]`)?.getAttribute('data-item-key'));
    });
    observer.observe(dialog, { attributes: true, childList: true, subtree: true });
    await pressKey(input, 'ArrowLeft');
    observer.disconnect();

    expect(activeIndex()).toBe(belowIndex - columns - 1);
    expect([...tileKeys].filter((key) => key !== belowKey)).toEqual([]);
    expect(dialog.querySelector(`[data-item-index="${belowIndex}"]`)?.getAttribute('data-item-key')).toBe(belowKey);
  });

  it('keeps a pointer-scrolled position when a later page changes the count', async () => {
    let total = 600;
    mocks.listItems.mockImplementation((filter: { offset: number }) => {
      const items = Array.from({ length: Math.max(0, Math.min(60, total - filter.offset)) }, (_, index) =>
        image(`scrolled-${filter.offset + index}.png`)
      );

      return { itemIndices: items.map((_, index) => filter.offset + index), items, offset: filter.offset, total };
    });

    const { dialog } = await openPicker();
    const firstOption = await vi.waitFor(() => {
      const option = dialog.querySelector<HTMLElement>('[role="option"]');

      expect(option).not.toBeNull();
      return option!;
    });
    let viewport: HTMLElement | null = firstOption.parentElement;

    while (viewport && viewport.scrollHeight <= viewport.clientHeight) {
      viewport = viewport.parentElement;
    }

    expect(viewport).not.toBeNull();
    // Another client adds an item, so pages fetched after the scroll report a larger count.
    total = 601;
    await act(() => {
      viewport!.scrollTop = viewport!.scrollHeight / 2;
      viewport!.dispatchEvent(new Event('scroll'));
    });
    const scrolledTop = viewport!.scrollTop;

    expect(scrolledTop).toBeGreaterThan(0);
    await vi.waitFor(() => expect(dialog.querySelector('[role="option"][aria-setsize="601"]')).not.toBeNull());
    await settle();

    expect(viewport!.scrollTop).toBe(scrolledTop);
  });

  it('keeps a pointer-scrolled position when an insert shifts the highlighted item', async () => {
    let inserted = 0;
    mocks.listItems.mockImplementation((filter: { offset: number }) => {
      const total = 600 + inserted;
      const items = Array.from({ length: Math.max(0, Math.min(60, total - filter.offset)) }, (_, index) => {
        const position = filter.offset + index - inserted;

        return image(position < 0 ? 'shifted-new.png' : `shifted-${position}.png`);
      });

      return { itemIndices: items.map((_, index) => filter.offset + index), items, offset: filter.offset, total };
    });

    const { dialog } = await openPicker();

    expect(getActiveOption(dialog)?.dataset.itemKey).toBe('image:shifted-0.png');
    let viewport: HTMLElement | null = getActiveOption(dialog)!.parentElement;

    while (viewport && viewport.scrollHeight <= viewport.clientHeight) {
      viewport = viewport.parentElement;
    }

    // Scroll the highlighted first row out of view while its page stays loaded.
    await act(() => {
      viewport!.scrollTop = getActiveOption(dialog)!.getBoundingClientRect().height * 4;
      viewport!.dispatchEvent(new Event('scroll'));
    });
    await settle();
    const scrolledTop = viewport!.scrollTop;

    expect(scrolledTop).toBeGreaterThan(0);
    inserted = 1;
    await act(() => queryClient!.invalidateQueries());
    await vi.waitFor(() => expect(dialog.querySelector('[role="option"][aria-setsize="601"]')).not.toBeNull());
    await settle();

    // The highlight moved one slot along with its item, which stays scrolled out of view.
    expect(getSearchInput(dialog).getAttribute('aria-activedescendant')).toMatch(/-slot-1$/);
    expect(viewport!.scrollTop).toBe(scrolledTop);
  });

  it('reveals an arrow move back onto the slot the highlight was first placed on before an insert shifted it', async () => {
    let inserted = 0;
    mocks.listItems.mockImplementation((filter: { offset: number }) => {
      const total = 600 + inserted;
      const items = Array.from({ length: Math.max(0, Math.min(60, total - filter.offset)) }, (_, index) => {
        const position = filter.offset + index - inserted;

        return image(position < 0 ? 'moved-new.png' : `moved-${position}.png`);
      });

      return { itemIndices: items.map((_, index) => filter.offset + index), items, offset: filter.offset, total };
    });

    const { dialog } = await openPicker();
    let viewport: HTMLElement | null = getActiveOption(dialog)!.parentElement;

    while (viewport && viewport.scrollHeight <= viewport.clientHeight) {
      viewport = viewport.parentElement;
    }

    inserted = 1;
    await act(() => queryClient!.invalidateQueries());
    await vi.waitFor(() => expect(dialog.querySelector('[role="option"][aria-setsize="601"]')).not.toBeNull());
    await settle();
    expect(getActiveOption(dialog)?.dataset.itemKey).toBe('image:moved-0.png');

    // Scroll the highlight out of view, then step back onto slot 0, where it was placed before the insert.
    await act(() => {
      viewport!.scrollTop = getActiveOption(dialog)!.getBoundingClientRect().height * 4;
      viewport!.dispatchEvent(new Event('scroll'));
    });
    await settle();
    expect(viewport!.scrollTop).toBeGreaterThan(0);

    await pressKey(getSearchInput(dialog), 'ArrowLeft');
    await settle();

    expect(getActiveOption(dialog)?.dataset.itemKey).toBe('image:moved-new.png');
    expect(viewport!.scrollTop).toBe(0);
  });

  it('keeps the highlight in view when the column count changes', async () => {
    const total = 60;
    mocks.listItems.mockImplementation((filter: { offset: number }) => {
      const items = Array.from({ length: Math.max(0, Math.min(60, total - filter.offset)) }, (_, index) =>
        image(`reflow-${filter.offset + index}.png`)
      );

      return { itemIndices: items.map((_, index) => filter.offset + index), items, offset: filter.offset, total };
    });

    const { dialog } = await openPicker();

    await pressKey(getSearchInput(dialog), 'End');
    await vi.waitFor(() => expect(getActiveOption(dialog)?.getAttribute('aria-posinset')).toBe(String(total)));
    const columnsBefore = getComputedStyle(getActiveOption(dialog)!.parentElement!).gridTemplateColumns;

    // Narrowing the popover drops a column, which moves the highlight's row further down.
    await act(() => {
      dialog.style.width = `${dialog.getBoundingClientRect().width * 0.7}px`;
    });
    await vi.waitFor(() =>
      expect(getComputedStyle(getActiveOption(dialog)!.parentElement!).gridTemplateColumns).not.toBe(columnsBefore)
    );
    await settle();

    const option = getActiveOption(dialog)!.getBoundingClientRect();
    let viewport: HTMLElement | null = getActiveOption(dialog)!.parentElement;

    while (viewport && viewport.scrollHeight <= viewport.clientHeight) {
      viewport = viewport.parentElement;
    }
    const bounds = viewport!.getBoundingClientRect();

    expect(option.top).toBeGreaterThanOrEqual(bounds.top - 1);
    expect(option.bottom).toBeLessThanOrEqual(bounds.bottom + 1);
  });

  it('shows and retries a failed uncached page within the current listing', async () => {
    const total = 120;
    let failedSecondPage = false;
    mocks.listItems.mockImplementation((filter: { offset: number }) => {
      if (filter.offset === 60 && !failedSecondPage) {
        failedSecondPage = true;
        return Promise.reject(new Error('page request failed'));
      }

      const items = Array.from({ length: Math.min(60, total - filter.offset) }, (_, index) =>
        image(`page-${filter.offset + index}.png`)
      );

      return {
        itemIndices: items.map((_, index) => filter.offset + index),
        items,
        offset: filter.offset,
        total,
      };
    });

    const { dialog } = await openPicker();
    const input = getSearchInput(dialog);

    await pressKey(input, 'End');
    await vi.waitFor(() =>
      expect(mocks.listItems.mock.calls.filter(([filter]) => filter.offset === 60)).toHaveLength(1)
    );
    const retry = await vi.waitFor(() => {
      const button = [...dialog.querySelectorAll<HTMLButtonElement>('button')].find((candidate) =>
        candidate.textContent?.includes('common.retry')
      );

      expect(button).toBeDefined();
      return button;
    });
    expect(retry).toBeDefined();
    const retryBounds = retry!.getBoundingClientRect();
    const dialogBounds = dialog.getBoundingClientRect();
    expect(retryBounds.top).toBeGreaterThanOrEqual(dialogBounds.top);
    expect(retryBounds.bottom).toBeLessThanOrEqual(dialogBounds.bottom);
    await act(() => retry!.click());
    expect(mocks.listItems.mock.calls.filter(([filter]) => filter.offset === 60)).toHaveLength(2);
  });

  it('closes on Escape and returns focus to the trigger', async () => {
    const { dialog, trigger } = await openPicker();

    await act(() =>
      getSearchInput(dialog).dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'Escape' }))
    );
    await settle();

    expect(document.querySelector(OPEN_DIALOG)).toBeNull();
    expect(onPick).not.toHaveBeenCalled();
    await vi.waitFor(() => expect(document.activeElement).toBe(trigger));
  });

  it('dims tiles the slot cannot take and ignores clicks on them', async () => {
    const { dialog } = await openPicker();
    const videoTile = getOption(dialog, 'video:c.mp4');

    expect(videoTile.getAttribute('aria-disabled')).toBe('true');
    expect(getComputedStyle(videoTile).opacity).toBe('0.35');

    await act(() => videoTile.click());

    expect(onPick).not.toHaveBeenCalled();
    expect(document.querySelector(OPEN_DIALOG)).not.toBeNull();
  });

  it('filters the board as you type and restarts the highlight at the first match', async () => {
    const { dialog } = await openPicker();

    await typeSearch(getSearchInput(dialog), 'cat');

    expect(mocks.listItems).toHaveBeenCalledWith(expect.objectContaining({ boardId: 'dogs', searchTerm: 'cat' }));
    expect(getOptions(dialog).map((option) => option.dataset.itemKey)).toEqual(['image:cat.png']);
    expect(getActiveOption(dialog)).toBe(getOption(dialog, 'image:cat.png'));
    expect(getStatus(dialog)).toBe('widgets.gallery.picker.matchCount');

    // With text in the field, Left/Right belong to the caret.
    const leftKey = new KeyboardEvent('keydown', { bubbles: true, cancelable: true, key: 'ArrowLeft' });

    await act(() => getSearchInput(dialog).dispatchEvent(leftKey));
    expect(leftKey.defaultPrevented).toBe(false);
    expect(getActiveOption(dialog)).toBe(getOption(dialog, 'image:cat.png'));
  });

  it('does not let Enter pick stale slots while a different board page is loading', async () => {
    let resolveCats:
      | ((page: { items: GalleryItem[]; itemIndices: number[]; offset: number; total: number }) => void)
      | null = null;

    mocks.listItems.mockImplementation((filter: { boardId: string; offset: number; searchTerm: string }) => {
      if (filter.boardId === 'cats') {
        return new Promise((resolve) => {
          resolveCats = resolve;
        });
      }

      return {
        itemIndices: dogItems.map((_, index) => index),
        items: dogItems,
        offset: filter.offset,
        total: dogItems.length,
      };
    });

    const { dialog } = await openPicker();
    const input = getSearchInput(dialog);

    await act(() => dialog.querySelector<HTMLButtonElement>('[aria-expanded]')?.click());
    await settle();
    await act(() => getBoardRow(dialog, 'Cats')?.click());
    await vi.waitFor(() => expect(mocks.listItems).toHaveBeenCalledWith(expect.objectContaining({ boardId: 'cats' })));

    await pressKey(input, 'Enter');

    expect(onPick).not.toHaveBeenCalled();
    expect(getOptions(dialog).some((option) => option.dataset.itemKey === 'image:b.png')).toBe(true);

    await act(() => resolveCats?.({ items: [image('cat.png', 'cats')], itemIndices: [0], offset: 0, total: 1 }));
    await settle();
    await vi.waitFor(() => expect(getOption(dialog, 'image:cat.png')).toBeDefined());
  });

  it('switches to the Assets view from the tabs', async () => {
    const { dialog } = await openPicker();
    // Icons label the tabs in the picker's narrow header; the view names stay their accessible names.
    const assetsTab = [...dialog.querySelectorAll<HTMLButtonElement>('[role="tab"]')].find((tab) =>
      tab.getAttribute('aria-label')?.startsWith('common.assets')
    );
    expect(assetsTab?.textContent).not.toContain('common.assets');

    await act(() => assetsTab?.click());
    await settle();

    expect(mocks.listItems).toHaveBeenCalledWith(expect.objectContaining({ boardId: 'dogs', galleryView: 'assets' }));
  });

  it('keeps multi-add pickers open until every slot is filled', async () => {
    const selection: GalleryPickerSelection = {
      addedKeys: new Set(['image:a.png']),
      mode: 'multiple',
      remaining: { image: 2 },
    };
    const { dialog } = await openPicker(<MultiAddHost initialSelection={selection} />);

    expect(dialog.querySelector('[role="listbox"]')?.getAttribute('aria-multiselectable')).toBe('true');
    expect(getOption(dialog, 'image:a.png').getAttribute('aria-disabled')).toBe('true');
    expect(getOption(dialog, 'image:a.png').getAttribute('aria-selected')).toBe('true');
    expect(getOption(dialog, 'image:b.png').getAttribute('aria-selected')).toBe('false');
    expect(getStatus(dialog)).toBe('widgets.gallery.picker.remainingCount · widgets.gallery.picker.itemCount');
    expect(dialog.textContent).toContain('common.done');

    await act(() => getOption(dialog, 'image:a.png').click());
    expect(onPick).not.toHaveBeenCalled();

    await act(() => getOption(dialog, 'image:b.png').click());

    expect(onPick).toHaveBeenCalledExactlyOnceWith(dogItems[1]);
    expect(document.querySelector(OPEN_DIALOG)).not.toBeNull();
    expect(getOption(dialog, 'image:b.png').getAttribute('aria-disabled')).toBe('true');
    expect(getOption(dialog, 'image:b.png').getAttribute('aria-selected')).toBe('true');

    await act(() => getOption(dialog, 'image:cat.png').click());

    expect(onPick).toHaveBeenCalledTimes(2);
    expect(document.querySelector(OPEN_DIALOG)).toBeNull();
  });

  it('switches boards from an in-place list and re-scopes the grid', async () => {
    const { dialog } = await openPicker();
    const input = getSearchInput(dialog);
    const boardButton = dialog.querySelector<HTMLButtonElement>('[aria-expanded]');

    await typeSearch(input, 'cat');
    await act(() => boardButton?.click());
    await settle();

    expect(boardButton?.getAttribute('aria-expanded')).toBe('true');
    expect(getOptions(dialog)).toHaveLength(0);
    expect(input.value).toBe('');
    expect(getStatus(dialog)).toBe('widgets.gallery.picker.boardCount');
    expect(getBoardRow(dialog, 'Dogs')?.getAttribute('aria-current')).toBe('true');
    expect(getBoardRow(dialog, 'Cats')).toBeDefined();

    // Enter with an empty search must not silently switch boards.
    await pressKey(input, 'Enter');
    expect(boardButton?.getAttribute('aria-expanded')).toBe('true');

    // ArrowDown hands focus to the first row; ArrowUp from it comes back.
    await pressKey(input, 'ArrowDown');
    const firstRow = getBoardsRegion(dialog)?.querySelector('button');

    expect(document.activeElement).toBe(firstRow);
    await pressKey(firstRow as HTMLElement, 'ArrowUp');
    expect(document.activeElement).toBe(input);

    await act(() => getBoardRow(dialog, 'widgets.gallery.uncategorized')?.click());
    await settle();

    expect(boardButton?.getAttribute('aria-expanded')).toBe('false');
    expect(mocks.listItems).toHaveBeenCalledWith(expect.objectContaining({ boardId: 'none' }));
    expect(getOptions(dialog).map((option) => option.dataset.itemKey)).toEqual(['image:loose.png']);
    expect(galleryCommands.selectBoard).not.toHaveBeenCalled();
  });

  // Assert video acceptance through the rendered picker; the regression was view wiring rather than the
  // accept-list builder.
  it('offers the full video and audio list on a video-kind picker', async () => {
    const { dialog } = await openPicker(
      <GalleryPickerPopover accept={['video']} label="Choose video" onPick={onPick}>
        <button type="button">Choose video</button>
      </GalleryPickerPopover>
    );
    const accept = dialog.querySelector<HTMLInputElement>('input[type="file"]')?.getAttribute('accept')?.split(',');

    expect(accept).toContain('video/*');
    expect(accept).toContain('audio/*');
    expect(accept).toEqual(getGalleryUploadAccept(['video']).split(','));
    expect(accept).not.toContain('image/png');
  });

  it('uploads into the current board and picks the result', async () => {
    const uploaded = {
      boardId: 'dogs',
      createdAt: '2026-09-09T00:00:00.000Z',
      height: 64,
      imageCategory: 'user' as const,
      imageName: 'fresh.png',
      imageUrl: '/full/fresh.png',
      queuedAt: '2026-09-09T00:00:00.000Z',
      sourceQueueItemId: 'queue-1',
      starred: false,
      thumbnailUrl: '/thumb/fresh.png',
      width: 64,
    };

    mocks.uploadGalleryImage.mockResolvedValue(uploaded);

    const { dialog } = await openPicker();
    const fileInput = dialog.querySelector<HTMLInputElement>('input[type="file"]');

    expect(fileInput?.getAttribute('accept')).toBe('image/png,image/jpeg,image/webp,.png,.jpg,.jpeg,.webp');
    expect(fileInput?.multiple).toBe(false);

    Object.defineProperty(fileInput, 'files', {
      configurable: true,
      value: [new File(['image'], 'fresh.png', { type: 'image/png' })],
    });
    await act(() => fileInput?.dispatchEvent(new Event('change', { bubbles: true })));
    await settle();

    expect(mocks.uploadGalleryImage).toHaveBeenCalledWith(expect.any(File), 'dogs', expect.anything());
    expect(onPick).toHaveBeenCalledExactlyOnceWith(legacyGeneratedImageToGalleryItem(uploaded));
    expect(onPick.mock.calls[0]?.[0]).toMatchObject({ fullUrl: '/full/fresh.png', kind: 'image', name: 'fresh.png' });
    expect(galleryCommands.selectItem).not.toHaveBeenCalled();
    expect(document.querySelector(OPEN_DIALOG)).toBeNull();
  });

  it('does not adopt an upload of a kind the slot cannot take', async () => {
    mocks.uploadGalleryVideo.mockResolvedValue(video('clip.mp4'));

    const { dialog } = await openPicker();
    const fileInput = dialog.querySelector<HTMLInputElement>('input[type="file"]');

    Object.defineProperty(fileInput, 'files', {
      configurable: true,
      value: [new File(['video'], 'clip.mp4', { type: 'video/mp4' })],
    });
    await act(() => fileInput?.dispatchEvent(new Event('change', { bubbles: true })));
    await settle();

    expect(mocks.uploadGalleryVideo).toHaveBeenCalledOnce();
    expect(onPick).not.toHaveBeenCalled();
    expect(document.querySelector(OPEN_DIALOG)).not.toBeNull();
  });

  it('hands off to the Gallery widget on the picker board', async () => {
    const { dialog } = await openPicker();

    await act(() => dialog.querySelector<HTMLButtonElement>('[aria-expanded]')?.click());
    await settle();
    await act(() => getBoardRow(dialog, 'widgets.gallery.uncategorized')?.click());
    await settle();

    const openButton = [...dialog.querySelectorAll<HTMLButtonElement>('button')].find((button) =>
      button.textContent?.includes('widgets.gallery.picker.openGallery')
    );

    await act(() => openButton?.click());

    expect(galleryCommands.selectBoard).toHaveBeenCalledExactlyOnceWith('none');
    expect(galleryCommands.setView).toHaveBeenCalledExactlyOnceWith('images');
    expect(openGallery).toHaveBeenCalledOnce();
    expect(document.querySelector(OPEN_DIALOG)).toBeNull();
  });

  it('leaves the Gallery board alone when it already matches, and stays open if the widget cannot open', async () => {
    openGallery.mockReturnValueOnce(false);

    const { dialog } = await openPicker();
    const openButton = [...dialog.querySelectorAll<HTMLButtonElement>('button')].find((button) =>
      button.textContent?.includes('widgets.gallery.picker.openGallery')
    );

    await act(() => openButton?.click());

    expect(galleryCommands.selectBoard).not.toHaveBeenCalled();
    expect(openGallery).toHaveBeenCalledOnce();
    expect(document.querySelector(OPEN_DIALOG)).not.toBeNull();
  });

  it('picks without a workbench and offers no hand-off to a Gallery widget there', async () => {
    const { dialog } = await openPicker(undefined, { withoutWorkbench: true });
    const [first] = getOptions(dialog);

    expect(first).toBeDefined();
    expect(
      [...dialog.querySelectorAll('button')].some((button) =>
        button.textContent?.includes('widgets.gallery.picker.openGallery')
      )
    ).toBe(false);

    await act(() => first!.click());

    expect(onPick).toHaveBeenCalledOnce();
    expect(document.querySelector(OPEN_DIALOG)).toBeNull();
  });

  it('reports a failed first listing with Retry instead of loading skeletons forever', async () => {
    mocks.listItems.mockImplementationOnce(() => {
      throw new Error('Service Unavailable');
    });
    const { dialog } = await openPicker();
    const input = getSearchInput(dialog);

    await vi.waitFor(() => expect(dialog.querySelector('[role="alert"]')?.textContent).toContain('listingLoadFailed'));
    expect(dialog.querySelector('[role="listbox"]')).toBeNull();
    expect(input.getAttribute('aria-controls')).toBeNull();
    // A count of a list that never loaded would read as an empty board.
    expect(getStatus(dialog)).toBe('');

    await pressKey(input, 'Enter');
    expect(onPick).not.toHaveBeenCalled();

    const retry = dialog.querySelector<HTMLButtonElement>('button[aria-label="widgets.gallery.retryLoadingItems"]');

    retry?.focus();
    await act(() => retry?.click());
    await settle();

    expect(dialog.querySelector('[role="alert"]')).toBeNull();
    expect(getOptions(dialog)).toHaveLength(6);
    // The Retry that held focus is gone; the search field, which drives the grid, takes it back.
    expect(document.activeElement).toBe(input);
  });

  it('shows a failed search as that search failing, not the previous results, and recovers with Retry', async () => {
    const { dialog } = await openPicker();
    const input = getSearchInput(dialog);

    expect(getOptions(dialog)).toHaveLength(6);
    mocks.listItems.mockImplementationOnce(() => {
      throw new Error('Service Unavailable');
    });
    await typeSearch(input, 'cat');

    await vi.waitFor(() => expect(dialog.querySelector('[role="alert"]')).not.toBeNull());
    expect(getOptions(dialog)).toEqual([]);
    expect(dialog.querySelector('[aria-busy="true"][role="listbox"]')).toBeNull();

    await pressKey(input, 'Enter');
    expect(onPick).not.toHaveBeenCalled();

    await act(() =>
      dialog.querySelector<HTMLButtonElement>('button[aria-label="widgets.gallery.retryLoadingItems"]')?.click()
    );
    await settle();

    expect(getOptions(dialog).map((option) => option.dataset.itemKey)).toEqual(['image:cat.png']);
    expect(mocks.listItems).toHaveBeenLastCalledWith(expect.objectContaining({ searchTerm: 'cat' }));
  });

  it('shows the previous scope only as a dimmed, inert placeholder while loading, and never during a retry', async () => {
    const { dialog } = await openPicker();
    const input = getSearchInput(dialog);
    let deliver: () => void = () => undefined;
    const deferCatResults = () =>
      new Promise((resolve) => {
        deliver = () => resolve({ items: [dogItems[3]], total: 1 });
      });

    mocks.listItems.mockImplementationOnce(deferCatResults);
    await typeSearch(input, 'cat');

    // The previous board's tiles stay on screen, but as a busy placeholder for the new search, not its results.
    const listbox = dialog.querySelector<HTMLElement>('[role="listbox"]');

    expect(listbox?.getAttribute('aria-busy')).toBe('true');
    expect(getOptions(dialog)).toHaveLength(6);
    expect(input.getAttribute('aria-activedescendant')).toBeNull();
    await act(() => getOption(dialog, 'image:a.png').click());
    await pressKey(input, 'Enter');
    expect(onPick).not.toHaveBeenCalled();
    expect(getStatus(dialog)).not.toBe('widgets.gallery.picker.matchCount');

    await act(() => deliver());
    await settle();

    expect(dialog.querySelector('[role="listbox"]')?.getAttribute('aria-busy')).toBeNull();
    expect(getOptions(dialog).map((option) => option.dataset.itemKey)).toEqual(['image:cat.png']);

    // A search that failed keeps showing its failure while it is retried; the placeholder does not return.
    mocks.listItems.mockImplementationOnce(() => {
      throw new Error('Service Unavailable');
    });
    await typeSearch(input, 'b');
    await vi.waitFor(() => expect(dialog.querySelector('[role="alert"]')).not.toBeNull());

    mocks.listItems.mockImplementationOnce(deferCatResults);
    await act(() =>
      dialog.querySelector<HTMLButtonElement>('button[aria-label="widgets.gallery.retryLoadingItems"]')?.click()
    );
    await settle();

    expect(dialog.querySelector('[role="alert"]')?.getAttribute('aria-busy')).toBe('true');
    expect(dialog.querySelector('[role="listbox"]')).toBeNull();
    expect(getOptions(dialog)).toEqual([]);

    await act(() => deliver());
    await settle();

    expect(dialog.querySelector('[role="alert"]')).toBeNull();
    expect(getOptions(dialog).map((option) => option.dataset.itemKey)).toEqual(['image:cat.png']);
  });
});
