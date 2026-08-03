/* oxlint-disable react-perf/jsx-no-new-object-as-prop */
import type { GalleryBoard } from '@features/gallery/core/types';

import { ChakraProvider } from '@chakra-ui/react';
import { DndContext } from '@dnd-kit/core';
import { DEFAULT_GALLERY_SETTINGS, type GallerySettings } from '@features/gallery/core/settings';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { GalleryStateView } from './galleryStateView';
import type { GalleryWidgetContextValue } from './GalleryWidgetContext';

import { GalleryBoardsPanel } from './GalleryBoardsPanel';
import { GalleryWidgetContext } from './GalleryWidgetContext';

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    i18n: { language: 'en' },
    t: (key: string, values?: Record<string, unknown>) => {
      if (key === 'widgets.gallery.createBoardNamed') {
        return `Create board "${String(values?.name)}"`;
      }
      if (key === 'widgets.gallery.boardGroups.boards') {
        return 'Your boards';
      }
      if (key === 'widgets.gallery.boardGroups.byDate') {
        return 'Dates';
      }
      if (key === 'common.archived') {
        return 'Archived';
      }
      if (key === 'widgets.gallery.uncategorized') {
        return 'Uncategorized';
      }

      return key;
    },
  }),
}));

const actions = {
  createBoard: vi.fn(async () => {}),
  selectBoard: vi.fn(),
  selectProjectBoard: vi.fn(async () => {}),
  updateSettings: vi.fn(),
};

const itemActions = { moveItemsToBoard: vi.fn(async () => {}) };

const createBoard = (overrides: Partial<GalleryBoard> & Pick<GalleryBoard, 'id' | 'name'>): GalleryBoard => ({
  archived: false,
  assetCount: 3,
  imageCount: 50,
  kind: 'board',
  videoCount: 0,
  ...overrides,
});

const boards = [
  createBoard({ id: 'dogs', name: 'dogs' }),
  createBoard({ id: 'cats', imageCount: 56, name: 'Cats', ownerName: 'Alice Example' }),
  createBoard({ assetCount: 1, id: 'none', imageCount: 1, kind: 'uncategorized', name: '' }),
  createBoard({ archived: true, id: 'gorl', imageCount: 1, name: 'GORL' }),
  createBoard({ id: 'by_date:2026-07-30', imageCount: 12, kind: 'date', name: 'Today' }),
];

const createGallery = (settings: Partial<GallerySettings> = {}): GalleryStateView =>
  ({
    boards,
    compareImageKey: null,
    currentItem: null,
    galleryView: 'images',
    isLoading: false,
    items: [],
    pendingPlaceholders: [],
    projectBoardId: null,
    searchTerm: '',
    selectedBoardId: 'dogs',
    selectedItemKey: null,
    selectedItemKeys: [],
    settings: { ...DEFAULT_GALLERY_SETTINGS, showArchivedBoards: true, showDateBoards: true, ...settings },
    ...({} as Record<string, never>),
  }) as GalleryStateView;

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const renderPanel = async (gallery: GalleryStateView = createGallery()) => {
  const contextValue = {
    actions,
    gallery,
    itemActions,
    projectName: 'Project',
  } as unknown as GalleryWidgetContextValue;

  await act(() =>
    root?.render(
      <ChakraProvider value={system}>
        <GalleryWidgetContext value={contextValue}>
          <DndContext>
            <GalleryBoardsPanel />
          </DndContext>
        </GalleryWidgetContext>
      </ChakraProvider>
    )
  );
};

const getBoardRows = (): HTMLElement[] =>
  Array.from(host?.querySelectorAll<HTMLElement>('[data-part="content"] button[type="button"]') ?? []);

const getSearchInput = (): HTMLInputElement => {
  const input = host?.querySelector<HTMLInputElement>('input');

  if (!input) {
    throw new Error('board search input did not render');
  }

  return input;
};

const type = async (input: HTMLInputElement, value: string) => {
  const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set;

  await act(async () => {
    setter?.call(input, value);
    input.dispatchEvent(new Event('input', { bubbles: true }));
    await Promise.resolve();
  });
};

const click = async (element: HTMLElement) => {
  await act(async () => {
    element.click();
    await Promise.resolve();
  });
};

beforeEach(() => {
  host = document.createElement('div');
  host.style.cssText = 'height:480px;left:20px;position:fixed;top:20px;width:280px;';
  document.body.append(host);
  root = createRoot(host);
  Object.values(actions).forEach((mock) => mock.mockClear());
  itemActions.moveItemsToBoard.mockClear();
});

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('GalleryBoardsPanel', () => {
  it('renders boards, dates, and archived as separate sections', async () => {
    await renderPanel();

    const text = host?.textContent ?? '';

    expect(text).toContain('Your boards');
    expect(text).toContain('Dates');
    expect(text).toContain('Archived');
    expect(text).toContain('GORL');
  });

  it('keeps archived boards out of the main list', async () => {
    await renderPanel();

    const sections = Array.from(host?.querySelectorAll('[data-scope="collapsible"]') ?? []);
    const boardsSection = sections[0];

    expect(boardsSection?.textContent).toContain('dogs');
    expect(boardsSection?.textContent).not.toContain('GORL');
  });

  it('marks the project board with a Project badge', async () => {
    // The fixture project used by the visual harness has no project board, so
    // this path is only reachable here.
    await renderPanel({ ...createGallery(), projectBoardId: 'cats' } as GalleryStateView);

    const catsRow = getBoardRows().find((row) => row.textContent?.includes('Cats'));

    expect(catsRow?.textContent).toContain('common.project');
  });

  it('shows media and asset counts together, so the row does not change meaning with the tab', async () => {
    await renderPanel();

    const catsRow = getBoardRows().find((row) => row.textContent?.includes('Cats'));

    // Cats has 56 images, 0 videos, 3 assets.
    expect(catsRow?.textContent).toContain('56 | 3');
  });

  it('does not spend row width on creation dates', async () => {
    await renderPanel();

    expect(host?.textContent).not.toMatch(/\b\d{1,2} [A-Z][a-z]{2}\b/);
  });

  it('marks the selected board for assistive tech', async () => {
    await renderPanel();

    const current = host?.querySelector('[aria-current="true"]');

    expect(current?.textContent).toContain('dogs');
  });

  it('renders owner subtitles only when the backend supplies an owner', async () => {
    await renderPanel();

    const catsRow = getBoardRows().find((row) => row.textContent?.includes('Cats'));
    const dogsRow = getBoardRows().find((row) => row.textContent?.includes('dogs'));

    expect(catsRow?.textContent).toContain('Alice Example');
    expect(dogsRow?.textContent).not.toContain('Alice Example');
  });

  it('keeps the owner subtitle truncated and changes it from muted to selected contrast styling', async () => {
    await renderPanel();

    const unselectedOwner = Array.from(host?.querySelectorAll<HTMLElement>('*') ?? []).find(
      (element) => element.childElementCount === 0 && element.textContent === 'Alice Example'
    );

    expect(unselectedOwner).toBeDefined();
    expect(getComputedStyle(unselectedOwner!).overflow).toBe('hidden');
    expect(getComputedStyle(unselectedOwner!).textOverflow).toBe('ellipsis');
    const unselectedColor = getComputedStyle(unselectedOwner!).color;

    await renderPanel({ ...createGallery(), selectedBoardId: 'cats' } as GalleryStateView);
    const selectedOwner = Array.from(host?.querySelectorAll<HTMLElement>('*') ?? []).find(
      (element) => element.childElementCount === 0 && element.textContent === 'Alice Example'
    );

    expect(selectedOwner).toBeDefined();
    expect(getComputedStyle(selectedOwner!).color).not.toBe(unselectedColor);
    expect(host?.querySelector('[aria-current="true"]')?.textContent).toContain('Alice Example');
  });

  it('shows the count for the active view and switches with the tab', async () => {
    await renderPanel();
    expect(host?.textContent).toContain('50');

    await renderPanel({ ...createGallery(), galleryView: 'assets' } as GalleryStateView);
    expect(host?.textContent).toContain('3');
  });

  it('selects a board and clears the search on click', async () => {
    await renderPanel();
    await type(getSearchInput(), 'cat');

    const catsRow = getBoardRows().find((row) => row.textContent?.includes('Cats'));

    await click(catsRow!);

    expect(actions.selectBoard).toHaveBeenCalledWith('cats');
    expect(getSearchInput().value).toBe('');
  });

  it('filters the list and offers to create an unmatched name', async () => {
    await renderPanel();
    await type(getSearchInput(), 'birds');

    expect(host?.textContent).toContain('Create board "birds"');

    const createRow = getBoardRows().find((row) => row.textContent?.includes('Create board'));

    await click(createRow!);

    expect(actions.createBoard).toHaveBeenCalledWith('birds');
  });

  it('creates on Enter only when nothing matched', async () => {
    await renderPanel();
    const input = getSearchInput();

    await type(input, 'dog');
    await act(async () => {
      input.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'Enter' }));
      await Promise.resolve();
    });
    expect(actions.createBoard).not.toHaveBeenCalled();

    await type(input, 'birds');
    await act(async () => {
      input.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'Enter' }));
      await Promise.resolve();
    });
    expect(actions.createBoard).toHaveBeenCalledWith('birds');
  });

  it('persists a collapsed section', async () => {
    await renderPanel();

    const trigger = host?.querySelector<HTMLElement>('[data-scope="collapsible"] [data-part="trigger"]');

    await click(trigger!);

    expect(actions.updateSettings).toHaveBeenCalledWith({ collapsedBoardSections: ['boards'] });
  });

  it('reopens a section that was persisted as collapsed', async () => {
    await renderPanel(createGallery({ collapsedBoardSections: ['boards'] }));

    const trigger = host?.querySelector<HTMLElement>('[data-scope="collapsible"] [data-part="trigger"]');

    await click(trigger!);

    expect(actions.updateSettings).toHaveBeenCalledWith({ collapsedBoardSections: [] });
  });

  it('toggles the date and archived board queries from the filter row', async () => {
    await renderPanel();

    const dateToggle = host?.querySelector<HTMLElement>('button[aria-label="widgets.gallery.hideDateBoards"]');

    await click(dateToggle!);
    expect(actions.updateSettings).toHaveBeenCalledWith({ showDateBoards: false });

    const archiveToggle = host?.querySelector<HTMLElement>('button[aria-label="widgets.gallery.hideArchivedBoards"]');

    await click(archiveToggle!);
    expect(actions.updateSettings).toHaveBeenCalledWith({ showArchivedBoards: false });
  });

  it('reports no matches when the only board named is one the filters are hiding', async () => {
    // Reachable while the archived toggle has flipped but the refetch that
    // drops archived boards has not landed: the name is taken, so creating is
    // refused, yet no section will show it.
    await renderPanel(createGallery({ showArchivedBoards: false }));
    await type(getSearchInput(), 'GORL');

    expect(host?.textContent).toContain('widgets.gallery.noBoardsMatchSearch');
    expect(host?.textContent).not.toContain('Create board');
  });
});
