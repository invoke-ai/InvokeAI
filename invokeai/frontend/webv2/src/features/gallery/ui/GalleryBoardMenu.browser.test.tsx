/* oxlint-disable react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import type { GalleryBoard, GalleryBoardAccess, GalleryBoardAction } from '@features/gallery/core/types';

import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { act, useCallback, useState } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import type { GalleryWidgetContextValue } from './GalleryWidgetContext';

import { GalleryBoardMenu } from './GalleryBoardMenu';
import { GalleryWidgetContext } from './GalleryWidgetContext';

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, values?: Record<string, unknown>) => {
      const messages: Record<string, string> = {
        'widgets.gallery.assetCount': `${String(values?.count)} uploads`,
        'common.cancel': 'Cancel',
        'widgets.gallery.deleteBoard': 'Delete board',
        'widgets.gallery.deleteBoardAndMedia': `Delete board and ${String(values?.count)} items`,
        'widgets.gallery.deleteBoardDescription': `Its ${String(values?.media)} move to Uncategorized.`,
        'widgets.gallery.deleteBoardEmptyDescription': 'The board is empty.',
        'widgets.gallery.deleteBoardIncludeMedia': `Also permanently delete its ${String(values?.count)} items`,
        'widgets.gallery.deleteBoardQuestion': `Delete board "${String(values?.name)}"?`,
        'widgets.gallery.deleteBoardTypeName': `Type "${String(values?.name)}" to confirm`,
        'widgets.gallery.downloadBoard': 'Download board',
        'widgets.gallery.downloadBoardWithOmission': `Download board (${String(values?.count)} video omitted)`,
        'widgets.gallery.imageCount': `${String(values?.count)} images`,
        'widgets.gallery.mediaCountsThree': `${String(values?.first)}, ${String(values?.second)} and ${String(
          values?.third
        )}`,
        'widgets.gallery.mediaCountsTwo': `${String(values?.first)} and ${String(values?.second)}`,
        'widgets.gallery.videoCount': `${String(values?.count)} videos`,
        'widgets.gallery.boardGroups.library': 'Library',
        'widgets.gallery.boardGroups.projects': 'Projects',
        'widgets.gallery.moveBoard': 'Move to…',
      };

      return messages[key] ?? key;
    },
  }),
}));

const board = {
  archived: false,
  assetCount: 0,
  assetVideoCount: 0,
  id: 'board-1',
  imageCount: 2,
  kind: 'board',
  name: 'Board 1',
  isInbox: false,
  projectId: null,
  videoCount: 1,
} as const;
const target = { board, x: 20, y: 20 };
const noop = vi.fn();
const context = {
  actions: {
    archiveBoard: vi.fn(),
    deleteBoard: vi.fn(),
    downloadBoard: vi.fn(),
    moveBoard: vi.fn(),
    renameBoard: vi.fn(),
    setBoardGeneralAccess: vi.fn(),
    updateSettings: vi.fn(),
  },
  gallery: {
    boards: [board],
    projectBoardId: null,
    selectedBoardId: 'board-1',
    settings: { autoAddBoardId: null, followSelectedBoard: false },
  },
  projectId: 'p1',
  projectName: 'Mahogany House',
  projectNames: new Map([
    ['p1', 'Mahogany House'],
    ['p2', 'Harbor Tower'],
  ]),
} as unknown as GalleryWidgetContextValue;
const targetContext = {
  ...context,
  gallery: { ...context.gallery, settings: { autoAddBoardId: 'board-1', followSelectedBoard: false } },
} as unknown as GalleryWidgetContextValue;

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

beforeEach(async () => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);

  await act(async () => {
    root?.render(
      <ChakraProvider value={system}>
        <GalleryWidgetContext value={context}>
          <GalleryBoardMenu target={target} onClose={noop} />
        </GalleryWidgetContext>
      </ChakraProvider>
    );
    await Promise.resolve();
  });
});

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

it('states the exact number of videos omitted from the image-only board archive', () => {
  expect(document.body.textContent).toContain('Download board (1 video omitted)');
});

it('offers a plain board download when the board holds no videos', async () => {
  await renderMenuFor({ ...board, videoCount: 0 } as GalleryBoard);

  expect(menuItemLabels()).toContain('Download board');
  expect(document.body.textContent).not.toContain('omitted');
});

const openDeleteDialog = async () => {
  const deleteItem = Array.from(document.querySelectorAll<HTMLElement>('[role="menuitem"]')).find(
    (element) => element.textContent === 'Delete board'
  );

  expect(deleteItem).toBeDefined();
  await act(() => userEvent.click(deleteItem!));
  await vi.waitFor(() => expect(document.querySelector('[role="alertdialog"]')).not.toBeNull());

  return document.querySelector<HTMLElement>('[role="alertdialog"]')!;
};

const dialogButton = (label: string) =>
  Array.from(document.querySelectorAll<HTMLButtonElement>('[role="alertdialog"] button')).find(
    (button) => button.textContent === label
  );

it('deletes the board and keeps its media unless the user also asks to delete the media', async () => {
  const dialog = await openDeleteDialog();

  // Only the kinds the board holds, and where they go.
  expect(dialog.textContent).toContain('Its 2 images and 1 videos move to Uncategorized.');
  expect(dialog.querySelector<HTMLInputElement>('input[type="checkbox"]')?.checked).toBe(false);
  await act(() => userEvent.click(dialogButton('Delete board')!));

  expect(context.actions.deleteBoard).toHaveBeenLastCalledWith('board-1', false);
});

it('deletes the media too only behind its checkbox, naming how many items go', async () => {
  const dialog = await openDeleteDialog();

  await act(() => userEvent.click(dialog.querySelector<HTMLElement>('[data-part="label"]')!));
  expect(dialog.textContent).toContain('Also permanently delete its 3 items');
  await act(() => userEvent.click(dialogButton('Delete board and 3 items')!));

  expect(context.actions.deleteBoard).toHaveBeenLastCalledWith('board-1', true);
});

it('asks for the board name before deleting many items with it', async () => {
  await renderMenuFor({ ...board, assetCount: 40, imageCount: 20 } as GalleryBoard);
  const dialog = await openDeleteDialog();

  await act(() => userEvent.click(dialog.querySelector<HTMLElement>('[data-part="label"]')!));
  const confirm = dialogButton('Delete board and 61 items')!;
  expect(confirm.disabled).toBe(true);

  await act(() => userEvent.type(dialog.querySelector<HTMLInputElement>('input:not([type="checkbox"])')!, 'Board 1'));
  expect(confirm.disabled).toBe(false);
  await act(() => userEvent.click(confirm));

  expect(context.actions.deleteBoard).toHaveBeenLastCalledWith('board-1', true);
});

/** The panel's part: the menu target clears when the menu closes. */
const ControlledMenu = ({ initialTarget }: { initialTarget: typeof target }) => {
  const [menuTarget, setMenuTarget] = useState<typeof target | null>(initialTarget);
  const close = useCallback(() => setMenuTarget(null), []);

  return <GalleryBoardMenu target={menuTarget} onClose={close} />;
};

const openMenus = () => document.querySelectorAll('[role="menu"][data-state="open"]');

it.each(['cancel', 'delete'] as const)(
  'closes the board menu as its delete dialog opens, and keeps it closed after %s',
  async (outcome) => {
    await act(async () => {
      root?.render(
        <ChakraProvider value={system}>
          <GalleryWidgetContext value={context}>
            <ControlledMenu initialTarget={target} />
          </GalleryWidgetContext>
        </ChakraProvider>
      );
      await Promise.resolve();
    });
    expect(openMenus()).toHaveLength(1);

    await openDeleteDialog();
    await vi.waitFor(() => expect(openMenus()).toHaveLength(0));

    await act(() => userEvent.click(dialogButton(outcome === 'cancel' ? 'Cancel' : 'Delete board')!));
    await vi.waitFor(() => expect(document.querySelector('[role="alertdialog"][data-state="open"]')).toBeNull());

    expect(openMenus()).toHaveLength(0);
    expect(document.activeElement?.closest('[role="menu"]')).toBeNull();
  }
);

it('offers no media choice for an empty board', async () => {
  await renderMenuFor({ ...board, imageCount: 0, videoCount: 0 } as GalleryBoard);
  const dialog = await openDeleteDialog();

  expect(dialog.textContent).toContain('The board is empty.');
  expect(dialog.querySelector('input[type="checkbox"]')).toBeNull();
});

it('makes the board the auto-add destination', async () => {
  const autoAddItem = document.querySelector<HTMLElement>('[role="menuitem"][data-value="auto-add-board"]');

  expect(autoAddItem?.textContent).toBe('widgets.gallery.autoAddToBoard');
  await act(() => userEvent.click(autoAddItem!));

  expect(context.actions.updateSettings).toHaveBeenCalledExactlyOnceWith({
    autoAddBoardId: 'board-1',
    followSelectedBoard: false,
  });
});

it('hands results back to the default destination from the auto-add board itself', async () => {
  await act(async () => {
    root?.render(
      <ChakraProvider value={system}>
        <GalleryWidgetContext value={targetContext}>
          <GalleryBoardMenu target={target} onClose={noop} />
        </GalleryWidgetContext>
      </ChakraProvider>
    );
    await Promise.resolve();
  });

  const item = document.querySelector<HTMLElement>('[role="menuitem"][data-value="auto-add-board"]');

  expect(item?.textContent).toBe('widgets.gallery.stopAutoAdd');
  await act(() => userEvent.click(item!));

  expect(context.actions.updateSettings).toHaveBeenLastCalledWith({ autoAddBoardId: null, followSelectedBoard: false });
});

it('offers neither choice on the default destination, where results already go', async () => {
  await renderMenuFor(board, {
    gallery: { ...context.gallery, projectBoardId: 'board-1' } as GalleryWidgetContextValue['gallery'],
  });

  expect(document.querySelector('[role="menuitem"][data-value="auto-add-board"]')).toBeNull();
});

const renderMenuFor = async (menuBoard: GalleryBoard, overrides: Partial<GalleryWidgetContextValue> = {}) => {
  const value = {
    ...context,
    gallery: { ...context.gallery, boards: [menuBoard] },
    ...overrides,
  } as GalleryWidgetContextValue;
  await act(async () => {
    root?.render(
      <ChakraProvider value={system}>
        <GalleryWidgetContext value={value}>
          <GalleryBoardMenu target={{ board: menuBoard, x: 20, y: 20 }} onClose={noop} />
        </GalleryWidgetContext>
      </ChakraProvider>
    );
    await Promise.resolve();
  });
};

const projectGroupElement = () =>
  Array.from(document.querySelectorAll<HTMLElement>('[role="group"]')).find(
    (element) => element.querySelector('[data-part="item-group-label"]')?.textContent === 'Projects'
  );

const projectGroup = () => {
  const group = projectGroupElement();

  return group
    ? {
        label: 'Projects',
        items: Array.from(group.querySelectorAll('[role="menuitem"]')).map((item) => item.textContent),
      }
    : undefined;
};

const libraryItem = () =>
  Array.from(document.querySelectorAll<HTMLElement>('[role="menuitem"]')).find(
    (element) => element.textContent === 'Library'
  );

const menuItemLabels = () =>
  Array.from(document.querySelectorAll<HTMLElement>('[role="menuitem"]')).map((item) => item.textContent);

it('offers every tier but the one the board is in, and moves it there', async () => {
  await renderMenuFor({ ...board, projectId: 'p1' } as GalleryBoard);

  const trigger = Array.from(document.querySelectorAll<HTMLElement>('[role="menuitem"]')).find(
    (element) => element.textContent === 'Move to…'
  )!;
  await act(() => userEvent.click(trigger));

  await vi.waitFor(() => {
    expect(menuItemLabels()).toContain('Harbor Tower');
  });
  // In a project: the Library and the other projects, never the project it is already in.
  expect(
    menuItemLabels().filter((label) => label === 'Library' || label === 'Harbor Tower' || label === 'Mahogany House')
  ).toEqual(['Library', 'Harbor Tower']);
  // The Library stands apart, above the projects' heading.
  expect(projectGroup()).toEqual({ label: 'Projects', items: ['Harbor Tower'] });
  expect(libraryItem()?.closest('[role="group"]')).toBeNull();
  expect(libraryItem()?.nextElementSibling?.getAttribute('role')).toBe('separator');

  const library = Array.from(document.querySelectorAll<HTMLElement>('[role="menuitem"]')).find(
    (element) => element.textContent === 'Library'
  )!;
  await act(() => userEvent.click(library));

  expect(context.actions.moveBoard).toHaveBeenCalledExactlyOnceWith('board-1', null, 'Library');
});

it('offers a Library board the open project first', async () => {
  await renderMenuFor(board);

  const trigger = Array.from(document.querySelectorAll<HTMLElement>('[role="menuitem"]')).find(
    (element) => element.textContent === 'Move to…'
  )!;
  await act(() => userEvent.click(trigger));

  await vi.waitFor(() => {
    expect(menuItemLabels()).toContain('Harbor Tower');
  });
  expect(
    menuItemLabels().filter((label) => label === 'Library' || label === 'Harbor Tower' || label === 'Mahogany House')
  ).toEqual(['Mahogany House', 'Harbor Tower']);
  expect(projectGroup()).toEqual({ label: 'Projects', items: ['Mahogany House', 'Harbor Tower'] });
  expect(libraryItem()).toBeUndefined();
  expect(projectGroupElement()?.previousElementSibling?.getAttribute('role')).not.toBe('separator');
});

it('leaves an inbox with only the actions its project does not own', async () => {
  await renderMenuFor({ ...board, isInbox: true, projectId: 'p1' } as GalleryBoard);

  const labels = menuItemLabels();

  expect(labels).toContain('widgets.gallery.exportProjectFromBoard');
  expect(labels).not.toContain('Move to…');
  expect(labels).not.toContain('Delete board');
  expect(labels).not.toContain('widgets.gallery.renameBoard');
});

const withAccess = (
  role: GalleryBoardAccess['role'],
  actions: GalleryBoardAction[],
  adminOverride = false
): GalleryBoard =>
  ({
    ...board,
    access: { actions, adminOverride, generalAccess: 'restricted', role },
    ownerName: 'Ana',
  }) as GalleryBoard;
const ALL_ACTIONS: GalleryBoardAction[] = [
  'read',
  'add_media',
  'manage_items',
  'rename',
  'set_cover',
  'archive',
  'set_general_access',
  'move',
  'delete',
];

it('offers a viewer only what they may do, and says whose board it is and whom to ask', async () => {
  await renderMenuFor(withAccess('viewer', ['read']), { boardSharing: true });

  expect(menuItemLabels()).toEqual(['Download board (1 video omitted)']);
  expect(document.body.textContent).toContain('widgets.gallery.boardAccess.header.viewer');
  expect(document.body.textContent).toContain('widgets.gallery.boardAccess.askToAdd');
});

it('lets an editor rename and archive someone else’s board, but not move, share or delete it', async () => {
  await renderMenuFor(withAccess('editor', ['read', 'add_media', 'manage_items', 'rename', 'set_cover', 'archive']), {
    boardSharing: true,
  });

  expect(menuItemLabels()).toEqual([
    'widgets.gallery.autoAddToBoard',
    'Download board (1 video omitted)',
    'widgets.gallery.renameBoard',
    'widgets.gallery.archiveBoard',
  ]);
  expect(document.body.textContent).toContain('widgets.gallery.boardAccess.header.editor');
  expect(document.body.textContent).not.toContain('askToAdd');
});

it('lets an owner change who can open the board, only where boards can be shared', async () => {
  await renderMenuFor(withAccess('owner', ALL_ACTIONS), { boardSharing: false });
  expect(menuItemLabels().some((label) => label?.includes('widgets.gallery.boardAccess.menu'))).toBe(false);
  expect(document.body.textContent).not.toContain('boardAccess.header');

  await renderMenuFor(withAccess('owner', ALL_ACTIONS), { boardSharing: true });
  const access = Array.from(document.querySelectorAll<HTMLElement>('[role="menuitem"]')).find((item) =>
    item.textContent?.includes('widgets.gallery.boardAccess.menu')
  )!;
  await act(() => userEvent.click(access));
  const everyone = await vi.waitFor(() => {
    const option = document.querySelector<HTMLElement>('[role="menuitemradio"][data-value="viewer"]');
    expect(option).not.toBeNull();
    return option!;
  });
  await act(() => userEvent.click(everyone));

  expect(context.actions.setBoardGeneralAccess).toHaveBeenCalledExactlyOnceWith('board-1', 'viewer');
});

it('gives an admin someone else’s board to manage, but no move into their own projects', async () => {
  await renderMenuFor(withAccess(null, ALL_ACTIONS, true), { boardSharing: true });

  const labels = menuItemLabels();
  expect(document.body.textContent).toContain('widgets.gallery.boardAccess.header.admin');
  expect(labels).toContain('Delete board');
  expect(labels).not.toContain('Move to…');
});

it('closes the rename dialog once the board no longer allows renaming', async () => {
  const editable = withAccess('editor', ['read', 'rename']);
  await renderMenuFor(editable);
  const rename = Array.from(document.querySelectorAll<HTMLElement>('[role="menuitem"]')).find(
    (item) => item.textContent === 'widgets.gallery.renameBoard'
  )!;
  await act(() => userEvent.click(rename));
  await vi.waitFor(() => expect(document.querySelector('[role="dialog"]')).not.toBeNull());

  // A refetch reports the role revoked to viewer.
  await renderMenuFor(editable, {
    gallery: { ...context.gallery, boards: [withAccess('viewer', ['read'])] } as GalleryWidgetContextValue['gallery'],
  });

  await vi.waitFor(() => expect(document.querySelector('[role="dialog"]')).toBeNull());

  // The role comes back on a later refetch: the dialog stays closed rather than reopening with what was typed.
  await renderMenuFor(editable, {
    gallery: { ...context.gallery, boards: [editable] } as GalleryWidgetContextValue['gallery'],
  });
  await new Promise((resolve) => {
    setTimeout(resolve, 100);
  });
  expect(document.querySelector('[role="dialog"]')).toBeNull();
});
