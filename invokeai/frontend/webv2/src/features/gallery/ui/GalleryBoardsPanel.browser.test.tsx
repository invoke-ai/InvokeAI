/* oxlint-disable react-perf/jsx-no-new-object-as-prop */
import type { GalleryBoard } from '@features/gallery/core/types';

import { ChakraProvider } from '@chakra-ui/react';
import { DndContext, KeyboardSensor, MouseSensor, pointerWithin, useSensor, useSensors } from '@dnd-kit/core';
import { DEFAULT_GALLERY_SETTINGS, type GallerySettings } from '@features/gallery/core/settings';
import { system } from '@theme/system';
import { act, type ReactNode } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import type { GalleryStateView } from './galleryStateView';
import type { GalleryWidgetContextValue } from './GalleryWidgetContext';

import { GalleryBoardDragMonitor } from './GalleryBoardDragMonitor';
import { GalleryBoardsPanel } from './GalleryBoardsPanel';
import { GalleryDragScope } from './galleryDnd';
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
        'widgets.gallery.createBoard': 'Create board',
        'widgets.gallery.createBoardIn': `Create board in ${String(values?.destination)}`,
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
  archiveBoard: vi.fn(() => Promise.resolve()),
  createBoard: vi.fn(() => Promise.resolve(true)),
  moveBoard: vi.fn(() => Promise.resolve()),
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
    primarySelectedItemKey: null,
    projectBoardId: 'mine',
    revealTargetPage: null,
    searchTerm: '',
    selectedBoardId: 'dogs',
    semanticImageQuery: null,
    semanticSearchText: null,
    selectedItemKey: null,
    selectedItemKeys: [],
    selectionStarredOnly: false,
    settings: { ...DEFAULT_GALLERY_SETTINGS, showArchivedBoards: true, showDateBoards: true, ...settings },
    starredOnly: false,
    ...({} as Record<string, never>),
  }) as GalleryStateView;

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

/** The workbench's drag context: the shell's mouse activation, pointer collisions, and the gallery's drop monitor. */
const WorkbenchDrag = ({ children }: { children: ReactNode }) => {
  const sensors = useSensors(
    useSensor(MouseSensor, { activationConstraint: { distance: 6 } }),
    useSensor(KeyboardSensor)
  );

  return (
    <DndContext collisionDetection={pointerWithin} sensors={sensors}>
      <GalleryDragScope value>
        {children}
        <GalleryBoardDragMonitor />
      </GalleryDragScope>
    </DndContext>
  );
};

const renderPanel = async (
  gallery: GalleryStateView = createGallery(),
  { currentUserId = null, withDrag = false }: { currentUserId?: string | null; withDrag?: boolean } = {}
) => {
  const contextValue = {
    actions,
    boardsState: { error: null, isRetrying: false, retry: () => Promise.resolve(), status: 'ready' },
    currentUserId,
    gallery,
    itemActions,
    projectId: 'p1',
    projectName: 'Mahogany House',
    projectNames: new Map([['p2', 'Harbor Tower']]),
    region: 'left',
  } as unknown as GalleryWidgetContextValue;

  await act(() =>
    root?.render(
      <ChakraProvider value={system}>
        <GalleryWidgetContext value={contextValue}>
          {withDrag ? (
            <WorkbenchDrag>
              <GalleryBoardsPanel />
            </WorkbenchDrag>
          ) : (
            <DndContext>
              <GalleryBoardsPanel />
            </DndContext>
          )}
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

const getCreateDialog = async () => {
  let dialog: HTMLElement | null = null;

  await vi.waitFor(() => {
    dialog = document.querySelector<HTMLElement>('[role="dialog"][data-state="open"]');
    expect(dialog).not.toBeNull();
  });

  return dialog!;
};
const getDialogInput = (dialog: HTMLElement) => dialog.querySelector<HTMLInputElement>('input[name="renameValue"]')!;
const submit = async (dialog: HTMLElement) => {
  await act(async () => {
    dialog.querySelector('form')!.requestSubmit();
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

  it('offers an unmatched name in both tiers, each row creating where it sits', async () => {
    await renderPanel();
    await type(getSearchInput(), 'birds');

    const sections = getRowsBySection();
    expect(sections['Mahogany House']).toEqual(['Create board "birds" in Mahogany House']);
    expect(sections.Library).toEqual(['Create board "birds" in Library']);

    await click(getBoardRows().find((row) => row.textContent?.includes('in Library'))!);
    expect(actions.createBoard).toHaveBeenCalledWith('birds', null);
    expect(getSearchInput().value).toBe('');

    await type(getSearchInput(), 'fish');
    await click(getBoardRows().find((row) => row.textContent?.includes('in Mahogany House'))!);
    expect(actions.createBoard).toHaveBeenLastCalledWith('fish', 'p1');
  });

  it('asks for a name in a dialog from a "+" and creates in that tier', async () => {
    await renderPanel();
    await click(getAddButton('widgets.gallery.createBoardInLibrary'));

    const dialog = await getCreateDialog();
    expect(dialog.querySelector('h2')?.textContent).toBe('Create board in Library');
    expect(document.activeElement).toBe(getDialogInput(dialog));
    expect(actions.createBoard).not.toHaveBeenCalled();

    await type(getDialogInput(dialog), 'birds');
    await submit(dialog);

    expect(actions.createBoard).toHaveBeenCalledExactlyOnceWith('birds', null);
    await vi.waitFor(() => expect(document.querySelector('[role="dialog"]')).toBeNull());

    await click(getAddButton('widgets.gallery.createBoardInProject'));
    expect((await getCreateDialog()).querySelector('h2')?.textContent).toBe('Create board in Mahogany House');
  });

  it('counts the boards in each tier, the inbox included, but not Uncategorized', async () => {
    await renderPanel();

    const counts = Array.from(
      host?.querySelectorAll<HTMLElement>('[data-scope="collapsible"][data-part="trigger"]') ?? []
    ).map((trigger) => trigger.textContent);

    expect(counts.slice(0, 2)).toEqual(['Mahogany House2', 'Library2']);
  });

  describe('dragging a board', () => {
    const row = (label: string) =>
      getBoardRows().find((button) => button.textContent?.startsWith(label))!.parentElement as HTMLElement;
    const heading = (label: string) =>
      [...(host?.querySelectorAll<HTMLElement>('[data-part="trigger"], [role="heading"]') ?? [])].find((element) =>
        element.textContent?.startsWith(label)
      )!;
    const settle = () =>
      new Promise<void>((resolve) => {
        globalThis.setTimeout(resolve, 30);
      });
    /**
     * A pointer that travels, as a hand does: the tiers start taking boards once the drag has begun, and are measured
     * on the moves after it. Outside act, as in the shell: dnd-kit settles the drop in effects act would hold back.
     */
    const drag = async (source: HTMLElement, target: HTMLElement) => {
      const center = (element: HTMLElement) => {
        const rect = element.getBoundingClientRect();

        return { x: rect.left + rect.width / 2, y: rect.top + rect.height / 2 };
      };
      const from = center(source);
      const to = center(target);
      const mouse = (type: string, at: { x: number; y: number }, element: EventTarget = document) =>
        element.dispatchEvent(new MouseEvent(type, { bubbles: true, button: 0, clientX: at.x, clientY: at.y }));

      mouse('mousedown', from, source);
      for (const progress of [0.25, 0.5, 0.75, 1, 1]) {
        mouse('mousemove', { x: from.x + (to.x - from.x) * progress, y: from.y + (to.y - from.y) * progress });
        await settle();
      }
      mouse('mouseup', to);
      await settle();
    };

    it('moves it to the project or Library it is dropped on, anywhere on that tier', async () => {
      await renderPanel(createGallery({ showOtherProjectBoards: true }), { withDrag: true });

      await drag(row('dogs'), heading('Harbor Tower'));
      expect(actions.moveBoard).toHaveBeenLastCalledWith('dogs', 'p2', 'Harbor Tower');

      await drag(row('Stripes'), row('Façades'));
      expect(actions.moveBoard).toHaveBeenLastCalledWith('theirs-member', 'p1', 'Mahogany House');

      await drag(row('Façades'), heading('Library'));
      expect(actions.moveBoard).toHaveBeenLastCalledWith('mine-member', null, 'Library');
      expect(actions.moveBoard).toHaveBeenCalledTimes(3);
    });

    it('moves nothing dropped on its own tier, and never drags an inbox, which moves with its project', async () => {
      await renderPanel(createGallery({ showOtherProjectBoards: true }), { withDrag: true });

      await drag(row('dogs'), row('Cats'));
      await drag(row('Inbox'), heading('Library'));
      // Pressing the row's menu button and moving away is no drag either.
      await drag(row('Façades').querySelector<HTMLElement>('.board-row-actions')!, heading('Library'));

      expect(actions.moveBoard).not.toHaveBeenCalled();
    });

    it('drags only boards the server would move, and only onto the caller’s own projects', async () => {
      // An admin sees other people's boards and projects; a shared board of their own stays in the Library.
      await renderPanel(
        {
          ...createGallery({ showOtherProjectBoards: true }),
          boards: boards.map((board) =>
            board.projectId === 'p2' || board.id === 'cats'
              ? { ...board, ownerId: 'them' }
              : board.id === 'dogs'
                ? { ...board, ownerId: 'me', visibility: 'shared' as const }
                : { ...board, ownerId: 'me' }
          ),
        } as GalleryStateView,
        { currentUserId: 'me', withDrag: true }
      );

      await drag(row('Cats'), heading('Mahogany House'));
      await drag(row('dogs'), heading('Mahogany House'));
      await drag(row('Façades'), heading('Harbor Tower'));
      expect(actions.moveBoard).not.toHaveBeenCalled();

      await drag(row('Façades'), heading('Library'));
      expect(actions.moveBoard).toHaveBeenCalledExactlyOnceWith('mine-member', null, 'Library');
    });

    it('takes a drop on a collapsed tier, by its heading', async () => {
      await renderPanel(createGallery({ collapsedBoardSections: ['library'] }), { withDrag: true });

      await drag(row('Façades'), heading('Library'));

      expect(actions.moveBoard).toHaveBeenCalledExactlyOnceWith('mine-member', null, 'Library');
    });

    it('selects a draggable board from the keyboard rather than starting a drag', async () => {
      await renderPanel(createGallery(), { withDrag: true });
      const button = getBoardRows().find((candidate) => candidate.textContent?.startsWith('Façades'))!;

      button.focus();
      await userEvent.keyboard('{Enter}');
      await userEvent.keyboard(' ');

      expect(actions.selectBoard).toHaveBeenCalledTimes(2);
      expect(actions.selectBoard).toHaveBeenLastCalledWith('mine-member');
      expect(actions.moveBoard).not.toHaveBeenCalled();
    });
  });

  describe('focus after a board moves', () => {
    it('follows a board archived from its menu into the Archived group', async () => {
      await renderPanel(galleryWithFacadesIn('p1'));
      const archive = await openMenuItem('widgets.gallery.archiveBoard');
      actions.archiveBoard.mockImplementationOnce(() => {
        void renderPanel(galleryWithFacadesIn('p1', true) as GalleryStateView);

        return Promise.resolve();
      });

      await act(() => userEvent.click(archive));

      await vi.waitFor(() => expect(document.activeElement).toBe(facadesButton()));
      expect(sectionOf(facadesButton())).toMatch(/^Archived/);
    });

    const galleryWithFacadesIn = (projectId: string | null, archived = false) => ({
      ...createGallery(),
      boards: boards.map((board) => (board.id === 'mine-member' ? { ...board, archived, projectId } : board)),
    });
    const menuItem = (label: string) =>
      [...document.querySelectorAll<HTMLElement>('[role="menuitem"]')].find((element) => element.textContent === label);
    const openMenuItem = async (label: string) => {
      await act(() => userEvent.click(facadesButton(), { button: 'right' }));

      return vi.waitFor(() => {
        expect(menuItem(label)).toBeDefined();
        return menuItem(label)!;
      });
    };
    const facadesButton = () => getBoardRows().find((button) => button.textContent?.startsWith('Façades'))!;
    const sectionOf = (element: Element) =>
      element.closest('[data-scope="collapsible"][data-part="root"]')?.querySelector('[data-part="trigger"]')
        ?.textContent;

    it('follows the board to its new tier', async () => {
      await renderPanel(galleryWithFacadesIn('p1'));
      facadesButton().focus();

      await renderPanel(galleryWithFacadesIn(null) as GalleryStateView);
      await act(() => Promise.resolve());

      expect(document.activeElement).toBe(facadesButton());
      expect(sectionOf(facadesButton())).toMatch(/^Library/);
    });

    it('follows a board moved from its menu once the menu has closed', async () => {
      await renderPanel(galleryWithFacadesIn('p1'));
      const menuItem = (label: string) =>
        [...document.querySelectorAll<HTMLElement>('[role="menuitem"]')].find(
          (element) => element.textContent === label
        );
      await act(() => userEvent.click(facadesButton(), { button: 'right' }));
      const moveTo = await vi.waitFor(() => {
        expect(menuItem('widgets.gallery.moveBoard')).toBeDefined();
        return menuItem('widgets.gallery.moveBoard')!;
      });
      await act(() => userEvent.click(moveTo));
      const library = await vi.waitFor(() => {
        const item = [...document.querySelectorAll<HTMLElement>('[role="menuitem"]')].find(
          (element) => element.textContent === 'Library'
        );
        expect(item).toBeDefined();
        return item!;
      });

      // As in the app, the move patches the board lists at once: the row leaves while the menu is still open.
      actions.moveBoard.mockImplementationOnce(() => {
        void renderPanel(galleryWithFacadesIn(null) as GalleryStateView);

        return Promise.resolve();
      });
      await act(() => userEvent.click(library));
      expect(actions.moveBoard).toHaveBeenCalledExactlyOnceWith('mine-member', null, 'Library');

      await vi.waitFor(() => expect(document.activeElement).toBe(facadesButton()));
      expect(sectionOf(facadesButton())).toMatch(/^Library/);
    });
  });

  it('names another project inbox for assistive tech while the visible row says Inbox', async () => {
    await renderPanel(createGallery({ showOtherProjectBoards: true }));

    const row = host?.querySelector<HTMLElement>('button[aria-label="Harbor Tower / Inbox"]');

    expect(row?.textContent).toContain('Inbox');
  });

  it('creates nothing from a dismissed dialog or a blank name, and leaves the search as it was', async () => {
    await renderPanel();
    await type(getSearchInput(), 'bir');
    await click(getAddButton('widgets.gallery.createBoardInProject'));

    const dialog = await getCreateDialog();
    await type(getDialogInput(dialog), '   ');
    await submit(dialog);
    expect(actions.createBoard).not.toHaveBeenCalled();
    await vi.waitFor(() => expect(document.querySelector('[role="dialog"]')).toBeNull());

    await click(getAddButton('widgets.gallery.createBoardInLibrary'));
    const reopened = await getCreateDialog();
    await click([...reopened.querySelectorAll('button')].find((button) => button.textContent === 'common.cancel')!);

    await vi.waitFor(() => expect(document.querySelector('[role="dialog"]')).toBeNull());
    expect(actions.createBoard).not.toHaveBeenCalled();
    expect(getSearchInput().value).toBe('bir');
  });

  it('keeps the dialog and its name when the create fails, so it can be tried again', async () => {
    actions.createBoard.mockResolvedValueOnce(false);
    await renderPanel();
    await type(getSearchInput(), 'birds');
    await click(getAddButton('widgets.gallery.createBoardInProject'));

    const dialog = await getCreateDialog();
    await submit(dialog);

    expect(actions.createBoard).toHaveBeenCalledExactlyOnceWith('birds', 'p1');
    expect(document.querySelector('[role="dialog"][data-state="open"]')).toBe(dialog);
    expect(getDialogInput(dialog).value).toBe('birds');
    expect(getSearchInput().value).toBe('birds');
  });

  it('starts the dialog from the typed search, so a name that found nothing is not retyped', async () => {
    await renderPanel();
    await type(getSearchInput(), 'birds');

    await click(getAddButton('widgets.gallery.createBoardInProject'));

    const dialog = await getCreateDialog();
    expect(getDialogInput(dialog).value).toBe('birds');
    expect(actions.createBoard).not.toHaveBeenCalled();

    await submit(dialog);

    expect(actions.createBoard).toHaveBeenCalledExactlyOnceWith('birds', 'p1');
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
