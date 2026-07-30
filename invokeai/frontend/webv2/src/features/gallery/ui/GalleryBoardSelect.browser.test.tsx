import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { BoardCover, BoardOptionContent } from './GalleryBoardSelect';

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, values?: Record<string, unknown>) => {
      if (key === 'widgets.gallery.imageCount') {
        return `${String(values?.count)} images`;
      }
      if (key === 'widgets.gallery.videoCount') {
        return `${String(values?.count)} videos`;
      }
      if (key === 'widgets.gallery.boardMediaCountBreakdown') {
        return `${String(values?.images)} · ${String(values?.videos)}`;
      }

      return key;
    },
  }),
}));

const board = {
  archived: false,
  assetCount: 2,
  coverImageName: null,
  coverThumbnailUrl: 'data:image/gif;base64,R0lGODlhAQABAAAAACw=',
  coverVideoName: 'clip.mp4',
  id: 'video-only',
  imageCount: 0,
  kind: 'board',
  name: 'Video only',
  videoCount: 5,
} as const;

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

beforeEach(() => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('board mixed-media presentation', () => {
  it('renders video-only boards with a mixed total, localized breakdown tooltip, and tabular numerals', async () => {
    await act(() =>
      root?.render(
        <ChakraProvider value={system}>
          <BoardOptionContent board={board} isSelected={false} />
        </ChakraProvider>
      )
    );

    await vi.waitFor(() => {
      expect(document.body.textContent).toContain('5 | 2');
    });

    const countBadge = Array.from(document.querySelectorAll('[aria-label]')).find(
      (element) => element.getAttribute('aria-label') === '0 images · 5 videos'
    );

    expect(countBadge).toBeDefined();
    expect(countBadge ? getComputedStyle(countBadge).fontVariantNumeric : '').toContain('tabular-nums');
    expect(countBadge?.getAttribute('data-scope')).toBe('tooltip');
    expect(countBadge?.getAttribute('data-part')).toBe('trigger');
  });

  it('overlays a decorative play glyph on static video covers', async () => {
    await act(() =>
      root?.render(
        <ChakraProvider value={system}>
          <BoardCover board={board} />
        </ChakraProvider>
      )
    );

    await vi.waitFor(() => {
      expect(document.querySelector('img')?.getAttribute('src')).toContain('data:image/gif');
    });
    expect(document.querySelector('svg[aria-hidden="true"]')).not.toBeNull();
    expect(document.querySelector('video')).toBeNull();
  });
});
