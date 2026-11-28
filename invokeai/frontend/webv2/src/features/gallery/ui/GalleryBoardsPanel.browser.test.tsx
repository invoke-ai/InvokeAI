/* oxlint-disable react-perf/jsx-no-new-object-as-prop */
import type { GalleryBoard } from '@features/gallery/core/types';

import { ChakraProvider } from '@chakra-ui/react';
import { DndContext, KeyboardSensor, MouseSensor, pointerWithin, useSensor, useSensors } from '@dnd-kit/core';
import { EMPTY_GALLERY_MARKS_FILTER } from '@features/gallery/core/marks';
import { DEFAULT_GALLERY_SETTINGS, type GallerySettings } from '@features/gallery/core/settings';
import { galleryBoardsOptions } from '@features/gallery/data/queries';
import { dragWithMouse } from '@platform/browser/mouseDrag.testing';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
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
    isCompareArmed: false,
    isSearchActive: false,
    items: [],
    listingBoardId: 'dogs',
    page: 0,
    pendingPlaceholders: [],
    primarySelectedItemKey: null,
    projectBoardId: 'mine',
    revealTargetPage: null,
    searchScope: 'board',
    searchTerm: '',
    selectedBoardId: 'dogs',
    semanticImageQuery: null,
    semanticSearchText: null,
    selectedItemKey: null,
    selectedItemKeys: [],
    selectionMarksFilter: EMPTY_GALLERY_MARKS_FILTER,
    settings: { ...DEFAULT_GALLERY_SETTINGS, showArchivedBoards: true, showDateBoards: true, ...settings },
    marksFilter: EMPTY_GALLERY_MARKS_FILTER,
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

const renderPanel = async (gallery: GalleryStateView = createGallery(), { withDrag = false } = {}) => {
  const contextValue = {
    actions,
    boardsState: { error: null, isRetrying: false, retry: () => Promise.resolve(), status: 'ready' },
    gallery,
    itemActions,
    projectId: 'p1',
    projectName: 'Mahogany House',
    projectNames: new Map([['p2', 'Harbor Tower']]),
    region: 'left',
  } as unknown as GalleryWidgetContextValue;

  // The listing submissions decide destinations from, as an earlier submission would have left it.
  const queryClient = new QueryClient();
  queryClient.setQueryData(galleryBoardsOptions({ includeArchived: true }).queryKey, gallery.boards);

  await act(() =>
    root?.render(
      <ChakraProvider value={system}>
        <QueryClientProvider client={queryClient}>
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
        </QueryClientProvider>
      </ChakraProvider>
    )
  );
};

const getBoardRows = (): HTMLElement[] =>
  Array.from(
    host?.querySelectorAll<HTMLElement>(
      '[data-scope="collapsible"][data-part="content"] button[type="button"]:not(.board-row-actions):not([data-gallery-project-group])'
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
        '[data-scope="collapsible"][data-part="content"] button[type="button"]:not(.board-row-actions):not([data-gallery-project-group])'
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

const AUTO_ADD_TARGET = '[role="img"][aria-label="widgets.gallery.autoAddTarget"]';

/** Labels of the rows marked as where new results go. */
const getAutoAddRows = (): string[] =>
  getBoardRows()
    .filter((row) => row.querySelector(AUTO_ADD_TARGET) !== null)
    .map((row) => (row.textContent ?? '').replace(/\d+ \| \d+$/, '').trim());

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
    // Nothing chosen: new results go to the project's inbox, and only its row says so.
    expect(getAutoAddRows()).toEqual(['Inbox']);
  });

  it('selects a board and clears the search on click', async () => {
    await renderPanel();
    await type(getSearchInput(), 'cat');

    const catsRow = getBoardRows().find((row) => row.textContent?.includes('Cats'));

    await click(catsRow!);

    expect(actions.selectBoard).toHaveBeenCalledWith('cats');
    expect(getSearchInput().value).toBe('');
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

  describe('dragging a board', () => {
    const row = (label: string) =>
      getBoardRows().find((button) => button.textContent?.startsWith(label))!.parentElement as HTMLElement;
    const heading = (label: string) =>
      [...(host?.querySelectorAll<HTMLElement>('[data-part="trigger"], [role="heading"]') ?? [])].find((element) =>
        element.textContent?.startsWith(label)
      )!;

    it('moves it to the project or Library it is dropped on, anywhere on that tier', async () => {
      await renderPanel(createGallery({ showOtherProjectBoards: true }), { withDrag: true });

      await dragWithMouse(row('dogs'), heading('Harbor Tower'));
      expect(actions.moveBoard).toHaveBeenLastCalledWith('dogs', 'p2', 'Harbor Tower');

      await dragWithMouse(row('Stripes'), row('Façades'));
      expect(actions.moveBoard).toHaveBeenLastCalledWith('theirs-member', 'p1', 'Mahogany House');

      await dragWithMouse(row('Façades'), heading('Library'));
      expect(actions.moveBoard).toHaveBeenLastCalledWith('mine-member', null, 'Library');
      expect(actions.moveBoard).toHaveBeenCalledTimes(3);
    });

    it("never drags someone else's board onto the caller's own tiers", async () => {
      const shared = createBoard({
        access: {
          actions: ['read', 'add_media', 'rename'],
          adminOverride: false,
          generalAccess: 'viewer',
          role: 'editor',
        },
        id: 'refs',
        name: 'Client references',
        ownerName: 'Cleo',
      });
      // An admin's rights allow the move, but its destinations would be the admin's projects, not the owner's.
      const adminOnly = createBoard({
        access: {
          actions: ['read', 'add_media', 'rename', 'move'],
          adminOverride: true,
          generalAccess: 'restricted',
          role: null,
        },
        id: 'sketches',
        name: 'Private sketches',
        ownerName: 'Dee',
      });
      await renderPanel(
        {
          ...createGallery({ expandedBoardSections: ['other-users'], showOtherProjectBoards: true }),
          boards: [...boards, shared, adminOnly],
        },
        { withDrag: true }
      );

      await dragWithMouse(row('Client references'), heading('Harbor Tower'));
      await dragWithMouse(row('Private sketches'), heading('Harbor Tower'));

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
  });

  it('lists other people’s boards apart, an admin’s view of them closed until opened', async () => {
    const shared = createBoard({
      access: { actions: ['read'], adminOverride: false, generalAccess: 'viewer', role: 'viewer' },
      id: 'refs',
      name: 'Client references',
      ownerName: 'Cleo',
    });
    const adminOnly = createBoard({
      access: { actions: ['read', 'add_media'], adminOverride: true, generalAccess: 'restricted', role: null },
      id: 'sketches',
      name: 'Private sketches',
      ownerName: 'Dee',
    });
    await renderPanel({ ...createGallery(), boards: [...boards, shared, adminOnly] });

    const sections = getRowsBySection();
    expect(sections['widgets.gallery.boardGroups.shared']).toEqual([
      'Client referenceswidgets.gallery.boardAccess.ownerAndAccess',
    ]);
    expect(sections.Library).not.toContain('Client references');
    const otherUsers = Array.from(
      host?.querySelectorAll<HTMLElement>('[data-scope="collapsible"][data-part="trigger"]') ?? []
    ).find((trigger) => trigger.textContent?.startsWith('widgets.gallery.boardGroups.otherUsers'));
    expect(otherUsers?.getAttribute('aria-expanded')).toBe('false');

    await click(otherUsers!);
    expect(actions.updateSettings).toHaveBeenCalledWith({ expandedBoardSections: ['other-users'] });
  });

  it('marks the inbox as the destination when the followed selected board is one the caller may only view', async () => {
    const shared = createBoard({
      access: { actions: ['read'], adminOverride: false, generalAccess: 'viewer', role: 'viewer' },
      id: 'refs',
      name: 'Client references',
      ownerName: 'Cleo',
    });
    await renderPanel({
      ...createGallery({ followSelectedBoard: true }),
      boards: [...boards, shared],
      selectedBoardId: 'refs',
    });

    // Only the inbox is marked, not the followed board that refuses results.
    expect(getAutoAddRows()).toEqual(['Inbox']);
  });

  it('persists a collapsed section', async () => {
    await renderPanel();

    const trigger = host?.querySelector<HTMLElement>('[data-scope="collapsible"] [data-part="trigger"]');

    await click(trigger!);

    expect(actions.updateSettings).toHaveBeenCalledWith({ collapsedBoardSections: ['project'] });
  });
});
