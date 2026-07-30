/* oxlint-disable react-perf/jsx-no-new-function-as-prop */
import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';

import type { GalleryWidgetContextValue } from './GalleryWidgetContext';

import { GalleryBoardMenu } from './GalleryBoardMenu';
import { GalleryWidgetContext } from './GalleryWidgetContext';

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string) =>
      (
        ({
          'widgets.gallery.downloadBoard': 'Download Board',
        }) as Record<string, string>
      )[key] ?? key,
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
