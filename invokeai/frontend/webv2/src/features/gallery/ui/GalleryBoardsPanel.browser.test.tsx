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
      const messages: Record<string, string> = {
        'common.archived': 'Archived',
        'widgets.gallery.boardGroups.byDate': 'Dates',
        'widgets.gallery.boardGroups.library': 'Library',
        'widgets.gallery.boardGroups.otherProjects': 'Other projects',
        'widgets.gallery.createBoardNamedIn': `Create board "${String(values?.name)}" in ${String(values?.destination)}`,
        'widgets.gallery.inbox': 'Inbox',
        'widgets.gallery.inboxOf': `${String(values?.project)} / Inbox`,
        'widgets.gallery.libraryBoardsHint': 'Boards here are available in every project.',
        'widgets.gallery.projectBoardsHint': 'Boards you create here belong to this project.',
        'widgets.gallery.uncategorized': 'Uncategorized',
      };

      return messages[key] ?? key;
    },
  }),
}));

const actions = {
  createBoard: vi.fn(async () => {}),
  selectBoard: vi.fn(),
  updateSettings: vi.fn(),
};

const itemActions = { moveItemsToBoard: vi.fn(async () => {}) };

const createBoard = (overrides: Partial<GalleryBoard> & Pick<GalleryBoard, 'id' | 'name'>): GalleryBoard => ({
  archived: false,
  assetCount: 3,
  assetVideoCount: 0,
  imageCount: 50,
  isInbox: false,
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
  createBoard({ id: 'mine-member', imageCount: 7, name: 'Façades', projectId: 'p1' }),
  createBoard({ id: 'mine', imageCount: 142, isInbox: true, name: 'Mahogany House', projectId: 'p1' }),
  createBoard({ id: 'theirs-member', name: 'Stripes', projectId: 'p2' }),
  createBoard({ id: 'theirs', isInbox: true, name: 'Harbor Tower', projectId: 'p2' }),
];

const createGallery = (settings: Partial<GallerySettings> = {}): GalleryStateView =>
  ({
    anchoredWindowPage: 0,
    boards,
    compareImageKey: null,
    galleryView: 'images',
    isComparisonActive: false,
    items: [],
    page: 0,
    pendingPlaceholders: [],
    projectBoardId: 'mine',
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
    boardsState: { error: null, isRetrying: false, retry: () => Promise.resolve(), status: 'ready' },
    gallery,
    itemActions,
    projectId: 'p1',
    projectName: 'Mahogany House',
    projectNames: new Map([['p2', 'Harbor Tower']]),
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
  Array.from(
    host?.querySelectorAll<HTMLElement>(
      '[data-scope="collapsible"][data-part="content"] button[type="button"]:not(.board-row-actions)'
    ) ?? []
  );

/** Row labels in document order, keyed by the section heading each sits under. */
const getRowsBySection = (): Record<string, string[]> => {
  const result: Record<string, string[]> = {};

  host?.querySelectorAll<HTMLElement>('[data-scope="collapsible"][data-part="root"]').forEach((section) => {
    const label = (section.querySelector('[data-part="trigger"]')?.textContent ?? '').replace(/\d+$/, '');

    // The row buttons only: not the section's own trigger, nor a row's trailing ⋮ control.
    result[label] = Array.from(
      section.querySelectorAll<HTMLElement>(
        '[data-scope="collapsible"][data-part="content"] button[type="button"]:not(.board-row-actions)'
      )
    ).map((row) =>
      (row.textContent ?? '')
        .replace(/\d+ \| \d+$/, '')
        .replace('Alice Example', '')
        .trim()
    );
  });

  return result;
};

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

const pressEnter = async (input: HTMLInputElement) => {
  await act(async () => {
    input.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'Enter' }));
    await Promise.resolve();
  });
};

const click = async (element: HTMLElement) => {
  await act(async () => {
    element.click();
    await Promise.resolve();
  });
};

const getAddButton = (label: string): HTMLButtonElement => {
  const button = host?.querySelector<HTMLButtonElement>(`button[aria-label="${label}"]`);

  if (!button) {
    throw new Error(`no "+" button labelled ${label}`);
  }

  return button;
};

beforeEach(() => {
  host = document.createElement('div');
  host.style.cssText = 'height:640px;left:20px;position:fixed;top:20px;width:520px;';
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
  it('groups the open project, the Library, dates and archived boards into their own sections', async () => {
    await renderPanel();

    const sections = getRowsBySection();

    expect(Object.keys(sections)).toEqual(['Mahogany House', 'Library', 'Dates', 'Archived']);
    // The inbox is pinned first and shown as the Inbox, not under the project's name.
    expect(sections['Mahogany House']).toEqual(['Inbox', 'Façades']);
    expect(sections.Library).toEqual(['Uncategorized', 'dogs', 'Cats']);
    expect(sections.Archived).toEqual(['GORL']);
  });

  it('lists other projects under their names only when they are shown', async () => {
    await renderPanel();
    expect(host?.textContent).not.toContain('Harbor Tower');

    await renderPanel(createGallery({ showOtherProjectBoards: true }));

    const sections = getRowsBySection();
    expect(Object.keys(sections)).toEqual(['Mahogany House', 'Library', 'Other projects', 'Dates', 'Archived']);
    expect(sections['Other projects']).toEqual(['Inbox', 'Stripes']);
    expect(host?.querySelector('[role="heading"][aria-level="4"]')?.textContent).toBe('Harbor Tower');
  });

  it('explains each tier once, while it holds only its fixed row', async () => {
    await renderPanel();
    expect(host?.textContent).not.toContain('belong to this project');
    expect(host?.textContent).not.toContain('available in every project');

    const onlyFixedRows = boards.filter((board) => board.isInbox || board.kind === 'uncategorized');
    await renderPanel({ ...createGallery(), boards: onlyFixedRows } as GalleryStateView);

    expect(host?.textContent).toContain('Boards you create here belong to this project.');
    expect(host?.textContent).toContain('Boards here are available in every project.');
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

  it('selects a board and clears the search on click', async () => {
    await renderPanel();
    await type(getSearchInput(), 'cat');

    const catsRow = getBoardRows().find((row) => row.textContent?.includes('Cats'));

    await click(catsRow!);

    expect(actions.selectBoard).toHaveBeenCalledWith('cats');
    expect(getSearchInput().value).toBe('');
  });

  it('creates an unmatched name in the open project by default', async () => {
    await renderPanel();
    await type(getSearchInput(), 'birds');

    const sections = getRowsBySection();
    expect(sections['Mahogany House']).toEqual(['Create board "birds" in Mahogany House']);
    expect(sections.Library).toEqual([]);

    await click(getBoardRows().find((row) => row.textContent?.includes('Create board'))!);

    expect(actions.createBoard).toHaveBeenCalledWith('birds', 'p1');
  });

  it('creates in the Library after its "+" chose that tier, and keeps that choice', async () => {
    await renderPanel();
    await click(getAddButton('widgets.gallery.createBoardInLibrary'));
    expect(document.activeElement).toBe(getSearchInput());

    await type(getSearchInput(), 'birds');
    expect(getRowsBySection().Library).toEqual(['Create board "birds" in Library']);

    await pressEnter(getSearchInput());
    expect(actions.createBoard).toHaveBeenCalledWith('birds', null);

    await type(getSearchInput(), 'fish');
    await pressEnter(getSearchInput());
    expect(actions.createBoard).toHaveBeenLastCalledWith('fish', null);

    await click(getAddButton('widgets.gallery.createBoardInProject'));
    await type(getSearchInput(), 'frogs');
    await pressEnter(getSearchInput());
    expect(actions.createBoard).toHaveBeenLastCalledWith('frogs', 'p1');
  });

  it('counts the boards made in each tier, not its fixed row', async () => {
    await renderPanel();

    const counts = Array.from(
      host?.querySelectorAll<HTMLElement>('[data-scope="collapsible"][data-part="trigger"]') ?? []
    ).map((trigger) => trigger.textContent);

    expect(counts.slice(0, 2)).toEqual(['Mahogany House1', 'Library2']);
  });

  it('names another project inbox for assistive tech while the visible row says Inbox', async () => {
    await renderPanel(createGallery({ showOtherProjectBoards: true }));

    const row = host?.querySelector<HTMLElement>('button[aria-label="Harbor Tower / Inbox"]');

    expect(row?.textContent).toContain('Inbox');
  });

  it('creates at once from a "+" when an unmatched name is already typed', async () => {
    await renderPanel();
    await type(getSearchInput(), 'birds');

    await click(getAddButton('widgets.gallery.createBoardInProject'));

    expect(actions.createBoard).toHaveBeenCalledWith('birds', 'p1');
    expect(getSearchInput().value).toBe('');
  });

  it('creates on Enter only when nothing matched', async () => {
    await renderPanel();
    const input = getSearchInput();

    await type(input, 'dog');
    await pressEnter(input);
    expect(actions.createBoard).not.toHaveBeenCalled();

    await type(input, 'birds');
    await pressEnter(input);
    expect(actions.createBoard).toHaveBeenCalledWith('birds', 'p1');
  });

  it('persists a collapsed section', async () => {
    await renderPanel();

    const trigger = host?.querySelector<HTMLElement>('[data-scope="collapsible"] [data-part="trigger"]');

    await click(trigger!);

    expect(actions.updateSettings).toHaveBeenCalledWith({ collapsedBoardSections: ['project'] });
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
