/* oxlint-disable react-perf/jsx-no-new-function-as-prop */
import { ChakraProvider } from '@chakra-ui/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { act, type ReactNode } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import type { GalleryWidgetContextValue } from './GalleryWidgetContext';

import { GalleryBoardMenu } from './GalleryBoardMenu';
import { GalleryUiProvider, type GalleryUiAdapter } from './GalleryUiContext';
import { GalleryWidgetContext } from './GalleryWidgetContext';

const mocks = vi.hoisted(() => ({ starredCount: 0 }));

vi.mock('@features/gallery/data/queries', () => ({
  galleryBoardStarredCountOptions: (boardId: string) => ({
    queryFn: () => Promise.resolve(mocks.starredCount),
    queryKey: ['board-starred-count', boardId],
  }),
}));

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
        'widgets.gallery.deleteBoardStarredNotice': `${String(values?.count)} starred kept`,
        'widgets.gallery.downloadBoardWithOmission': `Download Board (${String(values?.count)} video omitted)`,
        'widgets.gallery.imageCount': `${String(values?.count)} images`,
        'widgets.gallery.videoCount': `${String(values?.count)} videos`,
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
    renameBoard: vi.fn(),
    updateSettings: vi.fn(),
  },
  gallery: {
    projectBoardId: null,
    settings: { autoAddBoardId: 'follow' },
  },
} as unknown as GalleryWidgetContextValue;
const targetContext = {
  ...context,
  gallery: { ...context.gallery, settings: { autoAddBoardId: 'board-1' } },
} as unknown as GalleryWidgetContextValue;

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const adapters = {
  protected: { protectStarredMedia: true } as unknown as GalleryUiAdapter,
  unprotected: { protectStarredMedia: false } as unknown as GalleryUiAdapter,
};

const renderMenu = async (widgetContext: GalleryWidgetContextValue, protectStarredMedia = false) => {
  const adapter = adapters[protectStarredMedia ? 'protected' : 'unprotected'];
  const queryClient = new QueryClient();
  const tree: ReactNode = (
    <QueryClientProvider client={queryClient}>
      <GalleryUiProvider adapter={adapter}>
        <ChakraProvider value={system}>
          <GalleryWidgetContext value={widgetContext}>
            <GalleryBoardMenu target={target} onClose={noop} />
          </GalleryWidgetContext>
        </ChakraProvider>
      </GalleryUiProvider>
    </QueryClientProvider>
  );

  await act(async () => {
    root?.render(tree);
    await Promise.resolve();
  });
};

const openDeleteDialog = async () => {
  const deleteItem = Array.from(document.querySelectorAll('[role="menuitem"]')).find(
    (element) => element.textContent === 'Delete Board'
  );

  await act(() => userEvent.click(deleteItem as HTMLElement));
  await vi.waitFor(() => {
    expect(document.body.textContent).toContain('Delete Board and Media');
  });
};

beforeEach(async () => {
  mocks.starredCount = 0;
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);

  await renderMenu(context);
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
  await renderMenu(targetContext);

  const item = document.querySelector<HTMLElement>('[role="menuitem"][data-value="auto-add-board"]');

  expect(item?.textContent).toBe('widgets.gallery.stopAutoAdd');
  await act(() => userEvent.click(item!));

  expect(context.actions.updateSettings).toHaveBeenLastCalledWith({ autoAddBoardId: 'follow' });
});

it('says how many starred items stay when starred media is protected', async () => {
  mocks.starredCount = 3;
  await renderMenu(context, true);
  await openDeleteDialog();

  await vi.waitFor(() => {
    expect(document.body.textContent).toContain('3 starred kept');
  });
});

it('adds nothing to the dialog when starred media is not protected', async () => {
  mocks.starredCount = 3;
  await renderMenu(context, false);
  await openDeleteDialog();

  expect(document.body.textContent).not.toContain('starred kept');
});

it('adds nothing to the dialog when the board has no starred items', async () => {
  await renderMenu(context, true);
  await openDeleteDialog();
  await act(() => Promise.resolve());

  expect(document.body.textContent).not.toContain('starred kept');
});
