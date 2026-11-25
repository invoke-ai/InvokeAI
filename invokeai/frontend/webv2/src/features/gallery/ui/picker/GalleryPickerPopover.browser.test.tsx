/* oxlint-disable react-perf/jsx-no-new-object-as-prop, react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop */
import type { GalleryItem } from '@features/gallery/core/items';
import type { GalleryBoard } from '@features/gallery/core/types';
import type { GalleryItemContextMenuRequest, GalleryUiAdapter } from '@features/gallery/react';
import type { GalleryHost } from '@features/gallery/ui/GalleryUiContext';

import { ChakraProvider } from '@chakra-ui/react';
import { legacyGeneratedImageToGalleryItem } from '@features/gallery/core/items';
import { GalleryUiProvider } from '@features/gallery/react';
import { GalleryHostProvider } from '@features/gallery/ui/GalleryUiContext';
import { resolvedValues } from '@platform/core/resolvedValues.testing';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { act, useCallback, useState } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

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
  // The picker's board takes uploads; the destination rule has its own tests with the upload action.
  canAddToCachedGalleryBoard: () => true,
  ensureGalleryBoardDestinations: () => null,
  invalidateGalleryBoardDestinations: vi.fn(),
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
const openItemContextMenu = vi.fn();

const adapter = {
  gallery: galleryCommands,
  galleryValues: { galleryView: 'images', selectedBoardId: 'dogs', selectedImage: dogItems[1] },
  notifications,
  openItemContextMenu,
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

/** A host without a workbench (the Launchpad): no Gallery widget scope to seed from or reveal. */
const launchpadHost: GalleryHost = { galleryValues: resolvedValues(), notifications, projectName: '' };

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

  // The view is lazy-loaded on first open; wait for its actual ready/error UI and measured grid.
  const dialog = await vi.waitFor(
    () => {
      const opened = document.querySelector<HTMLElement>(OPEN_DIALOG);
      expect(opened).not.toBeNull();
      expect(opened!.querySelector('input:not([type="file"])')).not.toBeNull();
      expect(opened!.querySelector('[aria-busy="true"]')).toBeNull();
      expect(opened!.querySelector('[role="option"][data-item-key], [role="alert"]')).not.toBeNull();
      return opened!;
    },
    { timeout: 10_000 }
  );

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
    await vi.waitFor(() => expect(getOptions(dialog)).toHaveLength(6));
    await vi.waitFor(() => expect(getActiveOption(dialog)).toBe(getOption(dialog, 'image:b.png')));
    expect(getOption(dialog, 'image:b.png').getAttribute('aria-selected')).toBe('true');
    await vi.waitFor(() => expect(getStatus(dialog)).toBe('widgets.gallery.picker.itemCount'));

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
    await vi.waitFor(() => expect(getActiveOption(dialog)).toBe(getOption(dialog, 'image:a.png')));

    await pressKey(input, 'ArrowDown');
    await vi.waitFor(() => expect(getActiveOption(dialog)?.dataset.itemKey).toBe(`image:${ORDER[columns]}`));

    await pressKey(input, 'ArrowUp');
    await pressKey(input, 'ArrowRight');
    await pressKey(input, 'ArrowRight');
    await vi.waitFor(() => expect(getActiveOption(dialog)).toBe(getOption(dialog, 'video:c.mp4')));
    await vi.waitFor(() => expect(getStatus(dialog)).toBe('widgets.gallery.picker.unsupportedVideo'));

    await pressKey(input, 'Enter');
    expect(onPick).not.toHaveBeenCalled();

    await act(() => getOption(dialog, 'image:cat.png').click());
    expect(onPick).toHaveBeenCalledExactlyOnceWith(dogItems[3]);
    expect(document.querySelector(OPEN_DIALOG)).toBeNull();
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

    expect(document.querySelector(OPEN_DIALOG)).toBeNull();
    expect(onPick).not.toHaveBeenCalled();
    await vi.waitFor(() => expect(document.activeElement).toBe(trigger));
  });

  it("opens the highlighted item's menu from the keyboard in the search field, and leaves a right-click to the browser", async () => {
    const { dialog } = await openPicker();
    const input = getSearchInput(dialog);

    await vi.waitFor(() => expect(document.activeElement).toBe(input));
    await pressKey(input, 'ArrowLeft');
    await vi.waitFor(() => expect(getActiveOption(dialog)).toBe(getOption(dialog, 'image:a.png')));
    await act(() => userEvent.keyboard('{Shift>}{F10}{/Shift}'));

    const expectHighlightedItemMenu = () => {
      const request = openItemContextMenu.mock.lastCall?.[0] as GalleryItemContextMenuRequest;
      const tile = getOption(dialog, 'image:a.png').getBoundingClientRect();

      expect(request).toMatchObject({ item: dogItems[0], ref: { kind: 'image', name: 'a.png' } });
      expect(request.x).toBeGreaterThanOrEqual(tile.left);
      expect(request.x).toBeLessThanOrEqual(tile.right);
      expect(request.y).toBeGreaterThanOrEqual(tile.top);
      expect(request.y).toBeLessThanOrEqual(tile.bottom);
      expect(request.returnFocus?.()).toBe(input);
    };

    expectHighlightedItemMenu();

    await act(() => userEvent.click(input, { button: 'right' }));
    expect(openItemContextMenu).toHaveBeenCalledOnce();

    // A key-opened menu in an engine that reports it like a right-click: the key press came after the pointer's.
    const field = input.getBoundingClientRect();

    await act(() => userEvent.click(input));
    await pressKey(input, 'F10');
    await act(() =>
      input.dispatchEvent(
        new MouseEvent('contextmenu', {
          bubbles: true,
          button: 2,
          cancelable: true,
          clientX: field.left + field.width / 2,
          clientY: field.top + field.height / 2,
        })
      )
    );
    expect(openItemContextMenu).toHaveBeenCalledTimes(2);
    expectHighlightedItemMenu();
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
    await act(() => getBoardRow(dialog, 'Cats')?.click());
    await vi.waitFor(() => expect(mocks.listItems).toHaveBeenCalledWith(expect.objectContaining({ boardId: 'cats' })));

    await pressKey(input, 'Enter');

    expect(onPick).not.toHaveBeenCalled();
    expect(getOptions(dialog).some((option) => option.dataset.itemKey === 'image:b.png')).toBe(true);

    await act(() => resolveCats?.({ items: [image('cat.png', 'cats')], itemIndices: [0], offset: 0, total: 1 }));
    await vi.waitFor(() => expect(getOption(dialog, 'image:cat.png')).toBeDefined());
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
    await vi.waitFor(() =>
      expect(getStatus(dialog)).toBe('widgets.gallery.picker.remainingCount · widgets.gallery.picker.itemCount')
    );
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

  it('opens at the opening scope it is given instead of the Gallery widget scope', async () => {
    const { dialog } = await openPicker(
      <GalleryPickerPopover
        accept={['image']}
        label="Choose image"
        openingScope={{ boardId: 'none', galleryView: 'assets' }}
        onPick={onPick}
      >
        <button type="button">Choose image</button>
      </GalleryPickerPopover>
    );

    await vi.waitFor(() =>
      expect(getOptions(dialog).map((option) => option.dataset.itemKey)).toEqual(['image:loose.png'])
    );
    expect(mocks.listItems).toHaveBeenCalledWith(expect.objectContaining({ boardId: 'none', galleryView: 'assets' }));
    expect(mocks.listItems).not.toHaveBeenCalledWith(expect.objectContaining({ boardId: 'dogs' }));
    expect(dialog.querySelector('[role="tab"][aria-selected="true"]')?.getAttribute('aria-label')).toMatch(
      /^common\.assets/
    );
    expect(dialog.querySelector('[aria-expanded]')?.getAttribute('aria-label')).toBe(
      'widgets.gallery.picker.changeBoardNamed:widgets.gallery.uncategorized'
    );
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

    await vi.waitFor(() => expect(dialog.querySelector('[aria-busy="true"]')).toBeNull());
    expect(mocks.uploadGalleryVideo).toHaveBeenCalledOnce();
    expect(onPick).not.toHaveBeenCalled();
    expect(document.querySelector(OPEN_DIALOG)).not.toBeNull();
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
    await vi.waitFor(() => expect(getStatus(dialog)).toBe(''));

    await pressKey(input, 'Enter');
    expect(onPick).not.toHaveBeenCalled();

    const retry = dialog.querySelector<HTMLButtonElement>('button[aria-label="widgets.gallery.retryLoadingItems"]');

    retry?.focus();
    await act(() => retry?.click());

    await vi.waitFor(() => expect(getOptions(dialog)).toHaveLength(6));
    expect(dialog.querySelector('[role="alert"]')).toBeNull();
    // The Retry that held focus is gone; the search field, which drives the grid, takes it back.
    await vi.waitFor(() => expect(document.activeElement).toBe(input));
  });
});
