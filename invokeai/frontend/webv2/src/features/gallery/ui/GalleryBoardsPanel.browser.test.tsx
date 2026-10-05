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
  assetVideoCount: 0,
  imageCount: 50,
  kind: 'board',
  projectId: null,
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
    anchoredWindowPage: 0,
    boards,
    compareImageKey: null,
    galleryView: 'images',
    isComparisonActive: false,
    isLoading: false,
    items: [],
    page: 0,
    pendingPlaceholders: [],
    projectBoardId: null,
    revealTargetPage: null,
    searchTerm: '',
    selectedBoardId: 'dogs',
    semanticImageQuery: null,
    semanticSearchText: null,
    selectedItemKey: null,
    selectedItemKeys: [],
    settings: { ...DEFAULT_GALLERY_SETTINGS, showArchivedBoards: true, showDateBoards: true, ...settings },
    starredOnly: false,
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

  /** Badges and protection follow every board's owner; only hoisting depends on the active project. */
  it('marks every project-owned board with a Project badge', async () => {
    const ownedBoards = boards.map((board) => (board.id === 'cats' ? { ...board, projectId: 'p1' } : board));

    await renderPanel({ ...createGallery(), boards: ownedBoards, projectBoardId: 'cats' } as GalleryStateView);

    // The hoisted row shows the live project name, not the stored board name.
    const projectRow = getBoardRows().find((row) => row.textContent?.includes('Project'));

    expect(projectRow?.textContent).toContain('common.project');
  });

  it('shows media and asset counts together, so the row does not change meaning with the tab', async () => {
    await renderPanel();

    const catsRow = getBoardRows().find((row) => row.textContent?.includes('Cats'));

    expect(catsRow?.textContent).toContain('56 | 3');
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

  it('toggles every board group and the sort order from the one options menu', async () => {
    await renderPanel();

    const openMenu = async () => {
      const trigger = host?.querySelector<HTMLElement>('button[aria-label="widgets.gallery.filterAndSortBoards"]');

      await click(trigger!);
    };

    // Checkbox actions keep the menu open for multiple visibility and sort changes.
    await openMenu();

    const row = (value: string) => document.querySelector<HTMLElement>(`[data-scope="menu"] [data-value="${value}"]`);

    await click(row('date-boards')!);
    expect(actions.updateSettings).toHaveBeenCalledWith({ showDateBoards: false });

    await click(row('archived-boards')!);
    expect(actions.updateSettings).toHaveBeenCalledWith({ showArchivedBoards: false });

    await click(row('other-project-boards')!);
    expect(actions.updateSettings).toHaveBeenCalledWith({ showOtherProjectBoards: true });

    await click(row('board_name')!);
    expect(actions.updateSettings).toHaveBeenCalledWith({ boardOrderBy: 'board_name' });

    await click(row('ASC')!);
    expect(actions.updateSettings).toHaveBeenCalledWith({ boardOrderDir: 'ASC' });
  });
});
