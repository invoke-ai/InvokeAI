/* oxlint-disable react-perf/jsx-no-new-object-as-prop */
import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';

import type { GalleryWidgetContextValue } from './GalleryWidgetContext';

import { GalleryViewTabs } from './GalleryViewTabs';
import { GalleryWidgetContext } from './GalleryWidgetContext';

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

const toRgb = (color: string): [number, number, number] => {
  const context = document.createElement('canvas').getContext('2d')!;

  context.fillStyle = color;
  context.fillRect(0, 0, 1, 1);

  const [red, green, blue] = context.getImageData(0, 0, 1, 1).data;

  return [red!, green!, blue!];
};

const getRelativeLuminance = ([red, green, blue]: [number, number, number]): number => {
  const linearize = (channel: number): number => {
    const value = channel / 255;

    return value <= 0.03928 ? value / 12.92 : Math.pow((value + 0.055) / 1.055, 2.4);
  };

  return 0.2126 * linearize(red) + 0.7152 * linearize(green) + 0.0722 * linearize(blue);
};

/** Contrast of `foreground` at `alpha` composited over `background`. */
const getContrastRatio = (foreground: string, background: string, alpha: number): number => {
  const backgroundRgb = toRgb(background);
  const composited = toRgb(foreground).map((channel, index) =>
    Math.round(channel * alpha + backgroundRgb[index]! * (1 - alpha))
  ) as [number, number, number];
  const [lighter, darker] = [getRelativeLuminance(composited), getRelativeLuminance(backgroundRgb)].sort(
    (a, b) => b - a
  );

  return (lighter! + 0.05) / (darker! + 0.05);
};

const renderViewTabs = async (): Promise<{
  counts: HTMLElement[];
  indicator: HTMLElement;
  items: HTMLElement[];
}> => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);

  const value = {
    actions: { setView: vi.fn() },
    gallery: {
      boards: [{ assetCount: 7, id: 'board-1', imageCount: 148, videoCount: 0 }],
      galleryView: 'images',
      selectedBoardId: 'board-1',
    },
  } as unknown as GalleryWidgetContextValue;

  await act(() => {
    root?.render(
      <ChakraProvider value={system}>
        <GalleryWidgetContext value={value}>
          <GalleryViewTabs />
        </GalleryWidgetContext>
      </ChakraProvider>
    );
  });

  const items = [...host.querySelectorAll<HTMLElement>('[data-part="item"]')];

  return {
    // The count is the only span in an item whose text is purely numeric.
    counts: items.map((item) =>
      [...item.querySelectorAll<HTMLElement>('span')].find((span) => /^\d+$/.test(span.textContent!.trim()))!
    ),
    indicator: host.querySelector<HTMLElement>('[data-part="indicator"]')!,
    items,
  };
};

describe('GalleryViewTabs', () => {
  it('shows each view its own count', async () => {
    const { counts } = await renderViewTabs();

    expect(counts.map((count) => count.textContent)).toEqual(['148', '7']);
  });

  it('keeps the count readable on the selected segment', async () => {
    const { counts, indicator, items } = await renderViewTabs();
    const checkedIndex = items.findIndex((item) => item.dataset.state === 'checked');
    const count = counts[checkedIndex]!;
    const style = getComputedStyle(count);

    // The selected segment is filled with `accent.solid`, so a count pinned to
    // `fg.muted` (rather than dimmed from the item's own colour) is unreadable.
    const ratio = getContrastRatio(style.color, getComputedStyle(indicator).backgroundColor, Number(style.opacity));

    expect(ratio).toBeGreaterThanOrEqual(4.5);
  });
});
