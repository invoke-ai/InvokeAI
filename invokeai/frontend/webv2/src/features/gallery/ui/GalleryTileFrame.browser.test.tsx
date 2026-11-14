import type { GalleryItem } from '@features/gallery/core/items';

import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import {
  GALLERY_TILE_MEDIA_SELECTOR,
  GalleryTileFrame,
  type GalleryTileFrameProps,
  GalleryTileMedia,
  type GalleryTileMediaProps,
} from './GalleryTileFrame';

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const image: GalleryItem = {
  boardId: 'board',
  category: 'general',
  createdAt: '2026-09-01T00:00:00.000Z',
  fullUrl: '/full/a.png',
  height: 96,
  isIntermediate: false,
  kind: 'image',
  name: 'a.png',
  starred: false,
  thumbnailUrl: '/thumb/a.png',
  width: 128,
};

const video: GalleryItem = {
  boardId: 'board',
  category: 'general',
  createdAt: '2026-09-01T00:00:00.000Z',
  durationSeconds: 66,
  fullUrl: '/video/v.mp4',
  height: 96,
  isIntermediate: false,
  kind: 'video',
  name: 'v.mp4',
  starred: false,
  thumbnailUrl: '/video-thumb/v.mp4',
  width: 128,
};

const renderFrame = async (props: Partial<GalleryTileFrameProps> = {}, media: Partial<GalleryTileMediaProps> = {}) => {
  const item = props.item ?? image;

  await act(() =>
    root?.render(
      <ChakraProvider value={system}>
        <GalleryTileFrame data-testid="frame" item={item} {...props}>
          <GalleryTileMedia fit="square" height={item.height} isSelected={false} width={item.width} {...media} />
          {props.children}
        </GalleryTileFrame>
      </ChakraProvider>
    )
  );

  const frame = host?.querySelector<HTMLElement>('[data-testid="frame"]');

  if (!frame) {
    throw new Error('frame did not render');
  }

  return frame;
};

const mediaOf = (frame: HTMLElement) => {
  const media = frame.querySelector<HTMLElement>(GALLERY_TILE_MEDIA_SELECTOR);

  if (!media) {
    throw new Error('media box did not render');
  }

  return media;
};

const accentBorder = () => {
  const probe = document.createElement('div');

  probe.style.borderColor = 'var(--chakra-colors-accent-solid)';
  host?.appendChild(probe);

  const color = getComputedStyle(probe).borderTopColor;

  probe.remove();
  return color;
};

beforeEach(() => {
  host = document.createElement('div');
  document.body.appendChild(host);
  root = createRoot(host);
});

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('GalleryTileFrame', () => {
  it('flips the media border to accent when selected and keeps the shell props as invariants', async () => {
    const unselected = await renderFrame({ rounded: 'none', w: '10px' });

    expect(getComputedStyle(mediaOf(unselected)).borderTopColor).not.toBe(accentBorder());
    expect(getComputedStyle(unselected).aspectRatio).toBe('1 / 1');
    expect(getComputedStyle(unselected).width).not.toBe('10px');

    const selected = await renderFrame({}, { isSelected: true });

    expect(getComputedStyle(mediaOf(selected)).borderTopColor).toBe(accentBorder());
  });

  it('hides the dimensions badge until hovered, and composes caller css on top of its own', async () => {
    const frame = await renderFrame({ css: { opacity: 0.5 } }, { children: <button type="button">focus me</button> });
    const badge = frame.querySelector<HTMLElement>('.gallery-thumb-overlay');

    expect(badge?.textContent).toBe('128x96');
    expect(badge && getComputedStyle(badge).opacity).toBe('0');
    expect(getComputedStyle(frame).opacity).toBe('0.5');
    expect(getComputedStyle(mediaOf(frame)).outlineStyle).toBe('none');

    await act(() => frame.querySelector('button')?.focus());

    // The focus reveal comes from the frame's own css while the caller's opacity still applies, so the caller's css
    // was composed rather than substituted.
    // The reveal fades in.
    await vi.waitFor(() => expect(badge && getComputedStyle(badge).opacity).toBe('1'));
    expect(getComputedStyle(frame).opacity).toBe('0.5');
    // The focus ring is drawn on the media box, not the square cell.
    expect(getComputedStyle(frame).outlineStyle).toBe('none');
    expect(getComputedStyle(mediaOf(frame)).outlineWidth).toBe('2px');
    expect(getComputedStyle(mediaOf(frame)).outlineColor).toBe(accentBorder());
  });

  it('sizes the media box to the item ratio in aspect fit, clamped, and to the square otherwise', async () => {
    host!.style.width = '120px';
    const sizeOf = async (width: number, height: number, fit: 'aspect' | 'square' = 'aspect') => {
      const rect = mediaOf(await renderFrame({}, { fit, height, width })).getBoundingClientRect();

      return [rect.width, rect.height].map((value) => value.toFixed(1));
    };

    // Landscape fills the width; portrait narrows to ratio x width and keeps the cell's height.
    expect(await sizeOf(1920, 1080)).toEqual(['120.0', '67.5']);
    expect(await sizeOf(1024, 1536)).toEqual(['80.0', '120.0']);
    // Missing dimensions, and Fill mode, keep the square.
    expect(await sizeOf(0, 1536)).toEqual(['120.0', '120.0']);
    expect(await sizeOf(1024, 0)).toEqual(['120.0', '120.0']);
    expect(await sizeOf(1024, 1536, 'square')).toEqual(['120.0', '120.0']);
    // A panorama stops at 3:1 rather than shrinking to a sliver.
    expect(await sizeOf(8000, 1000)).toEqual(['120.0', '40.0']);
    expect(await sizeOf(1000, 8000)).toEqual(['40.0', '120.0']);
  });

  it('pins the dimensions badge when asked', async () => {
    const frame = await renderFrame({ alwaysShowDimensions: true });
    const badge = frame.querySelector<HTMLElement>('.gallery-thumb-overlay');

    expect(badge && getComputedStyle(badge).opacity).toBe('1');
  });

  it('shows a duration badge for videos instead of dimensions', async () => {
    const frame = await renderFrame({ item: video });

    expect(frame.querySelector('.gallery-thumb-overlay')).toBeNull();
    expect(frame.textContent).toContain('1:06');
    expect(frame.querySelector('svg.lucide-play[aria-hidden="true"]')).not.toBeNull();
  });
});
