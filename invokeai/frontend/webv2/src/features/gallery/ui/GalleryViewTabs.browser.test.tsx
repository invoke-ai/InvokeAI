/* oxlint-disable react-perf/jsx-no-new-object-as-prop */
import { ChakraProvider } from '@chakra-ui/react';
import { getContrastRatio } from '@platform/ui/theme/contrastRatio.testing';
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

  // Match the release accessibility journey: wait until Ark has measured the
  // moving indicator and removed its initial checked-item fallback styling.
  await act(
    () =>
      new Promise<void>((resolve) => {
        requestAnimationFrame(() =>
          requestAnimationFrame(() => {
            setTimeout(resolve, 200);
          })
        );
      })
  );

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

  it('keeps the count readable against the selected segment itself', async () => {
    const { counts, items } = await renderViewTabs();
    const checkedIndex = items.findIndex((item) => item.dataset.state === 'checked');
    const count = counts[checkedIndex]!;
    const style = getComputedStyle(count);

    // Axe evaluates the selected item's own background rather than a moving
    // indicator sibling behind it, so that local surface must carry the fill.
    const ratio = getContrastRatio(
      style.color,
      getComputedStyle(items[checkedIndex]!).backgroundColor,
      Number(style.opacity)
    );

    expect(ratio).toBeGreaterThanOrEqual(4.5);
  });
});
