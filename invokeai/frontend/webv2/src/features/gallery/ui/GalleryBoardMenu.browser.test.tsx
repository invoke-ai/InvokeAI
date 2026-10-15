/* oxlint-disable react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import type { GalleryBoard } from '@features/gallery/core/types';

import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { act } from 'react';
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
        'widgets.gallery.assetCount': `${String(values?.count)} assets`,
        'common.cancel': 'Cancel',
        'widgets.gallery.boardItemCounts': `${String(values?.images)} · ${String(values?.videos)} · ${String(
          values?.assets
        )}`,
        'widgets.gallery.deleteBoard': 'Delete Board',
        'widgets.gallery.deleteBoardAndMedia': 'Delete Board and Media',
        'widgets.gallery.deleteBoardDescription': 'Choose whether the board media moves or is deleted.',
        'widgets.gallery.deleteBoardOnly': 'Delete Board Only',
        'widgets.gallery.deleteBoardQuestion': `Delete board "${String(values?.name)}"?`,
        'widgets.gallery.downloadBoardWithOmission': `Download Board (${String(values?.count)} video omitted)`,
        'widgets.gallery.imageCount': `${String(values?.count)} images`,
        'widgets.gallery.videoCount': `${String(values?.count)} videos`,
        'widgets.gallery.boardGroups.library': 'Library',
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
    updateSettings: vi.fn(),
  },
  gallery: {
    projectBoardId: null,
    settings: { autoAddBoardId: 'follow' },
  },
  // Signed in, as an admin can be, alongside boards that are not theirs.
  currentUserId: 'me',
  projectId: 'p1',
  projectName: 'Mahogany House',
  projectNames: new Map([
    ['p1', 'Mahogany House'],
    ['p2', 'Harbor Tower'],
  ]),
} as unknown as GalleryWidgetContextValue;
const targetContext = {
  ...context,
  gallery: { ...context.gallery, settings: { autoAddBoardId: 'board-1' } },
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
  expect(document.body.textContent).toContain('Download Board (1 video omitted)');
});

it('shows image, video, and asset counts before deleting a mixed-media board', async () => {
  const deleteItem = Array.from(document.querySelectorAll('[role="menuitem"]')).find(
    (element) => element.textContent === 'Delete Board'
  );

  expect(deleteItem).toBeDefined();
  await act(() => userEvent.click(deleteItem as HTMLElement));

  await vi.waitFor(() => {
    expect(document.body.textContent).toContain('2 images · 1 videos · 0 assets');
  });
  expect(document.body.textContent).toContain('Delete Board and Media');
});

it('makes the board the auto-add destination', async () => {
  const autoAddItem = document.querySelector<HTMLElement>('[role="menuitem"][data-value="auto-add-board"]');

  expect(autoAddItem?.textContent).toBe('widgets.gallery.autoAddToBoard');
  await act(() => userEvent.click(autoAddItem!));

  expect(context.actions.updateSettings).toHaveBeenCalledExactlyOnceWith({ autoAddBoardId: 'board-1' });
});

it('hands results back to the selected board from the auto-add board itself', async () => {
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

  expect(context.actions.updateSettings).toHaveBeenLastCalledWith({ autoAddBoardId: 'follow' });
});

const renderMenuFor = async (menuBoard: GalleryBoard, overrides: Partial<GalleryWidgetContextValue> = {}) => {
  await act(async () => {
    root?.render(
      <ChakraProvider value={system}>
        <GalleryWidgetContext value={{ ...context, ...overrides }}>
          <GalleryBoardMenu target={{ board: menuBoard, x: 20, y: 20 }} onClose={noop} />
        </GalleryWidgetContext>
      </ChakraProvider>
    );
    await Promise.resolve();
  });
};

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
});

it.each([
  ['someone else’s board, which only its owner can move into their projects', { ownerId: 'them' }],
  ['a shared or public board, which stays in the Library', { ownerId: 'me', visibility: 'shared' as const }],
])('offers no move for %s, while keeping the rest of its menu', async (_label, overrides) => {
  await renderMenuFor({ ...board, ...overrides } as GalleryBoard);

  const labels = menuItemLabels();

  expect(labels).not.toContain('Move to…');
  expect(labels).toContain('widgets.gallery.renameBoard');
});

it('offers the move on a single-user install, where every board is the one user’s whatever owner it records', async () => {
  await renderMenuFor({ ...board, ownerId: 'system' } as GalleryBoard, { currentUserId: null });

  expect(menuItemLabels()).toContain('Move to…');
});

it('leaves an inbox with only the actions its project does not own', async () => {
  await renderMenuFor({ ...board, isInbox: true, projectId: 'p1' } as GalleryBoard);

  const labels = menuItemLabels();

  expect(labels).toContain('widgets.gallery.exportProjectFromBoard');
  expect(labels).not.toContain('Move to…');
  expect(labels).not.toContain('Delete Board');
  expect(labels).not.toContain('widgets.gallery.renameBoard');
});
