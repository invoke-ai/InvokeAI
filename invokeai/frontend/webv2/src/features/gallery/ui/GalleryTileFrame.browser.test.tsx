import type { GalleryItem } from '@features/gallery/core/items';

import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import { GalleryTileFrame, type GalleryTileFrameProps } from './GalleryTileFrame';

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

const renderFrame = async (props: Partial<GalleryTileFrameProps> = {}) => {
  await act(() =>
    root?.render(
      <ChakraProvider value={system}>
        <GalleryTileFrame data-testid="frame" item={image} {...props} />
      </ChakraProvider>
    )
  );

  const frame = host?.querySelector<HTMLElement>('[data-testid="frame"]');

  if (!frame) {
    throw new Error('frame did not render');
  }

  return frame;
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
  it('flips the border to accent when selected and keeps the shell props as invariants', async () => {
    const unselected = await renderFrame({ rounded: 'none', w: '10px' });

    expect(getComputedStyle(unselected).borderTopColor).not.toBe(accentBorder());
    expect(getComputedStyle(unselected).aspectRatio).toBe('1 / 1');
    expect(getComputedStyle(unselected).width).not.toBe('10px');

    const selected = await renderFrame({ isSelected: true });

    expect(getComputedStyle(selected).borderTopColor).toBe(accentBorder());
  });

  it('hides the dimensions badge until hovered, and composes caller css on top of its own', async () => {
    const frame = await renderFrame({
      children: <button type="button">focus me</button>,
      css: { opacity: 0.5 },
    });
    const badge = frame.querySelector<HTMLElement>('.gallery-thumb-overlay');

    expect(badge?.textContent).toBe('128x96');
    expect(badge && getComputedStyle(badge).opacity).toBe('0');
    expect(getComputedStyle(frame).opacity).toBe('0.5');
    expect(getComputedStyle(frame).outlineStyle).toBe('none');

    await act(() => frame.querySelector('button')?.focus());

    // The focus ring comes from the frame's own css, so it proves the caller's
    // css was composed rather than substituted.
    expect(getComputedStyle(frame).outlineWidth).toBe('2px');
    expect(getComputedStyle(frame).outlineColor).toBe(accentBorder());
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
