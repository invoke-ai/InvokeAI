/* oxlint-disable react-perf/jsx-no-new-function-as-prop */
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
      };

      return messages[key] ?? key;
    },
  }),
}));

const board = {
  archived: false,
  assetCount: 0,
  id: 'board-1',
  imageCount: 2,
  kind: 'board',
  name: 'Board 1',
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
  },
  gallery: {
    projectBoardId: null,
  },
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
