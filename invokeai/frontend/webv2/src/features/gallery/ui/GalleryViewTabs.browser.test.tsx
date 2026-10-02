/* oxlint-disable react-perf/jsx-no-new-object-as-prop */
import { ChakraProvider } from '@chakra-ui/react';
import { compositeColors, getContrastRatio } from '@platform/ui/theme/contrastRatio.testing';
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
  tabs: HTMLElement[];
}> => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);

  const value = {
    actions: { setView: vi.fn() },
    gallery: {
      boards: [{ assetCount: 7, assetVideoCount: 0, id: 'board-1', imageCount: 148, videoCount: 0 }],
      galleryView: 'images',
      selectedBoardId: 'board-1',
    },
  } as unknown as GalleryWidgetContextValue;

  await act(() => {
    root?.render(
      <ChakraProvider value={system}>
        <GalleryWidgetContext value={value}>
          <GalleryViewTabs idBase="gallery-view-tabs-test" />
        </GalleryWidgetContext>
      </ChakraProvider>
    );
  });

  const tabs = [...host.querySelectorAll<HTMLElement>('[role="tab"]')];

  return {
    // The count is the only span in a tab whose text is purely numeric.
    counts: tabs.map((tab) =>
      [...tab.querySelectorAll<HTMLElement>('span')].find((span) => /^\d+$/.test(span.textContent!.trim()))!
    ),
    tabs,
  };
};

describe('GalleryViewTabs', () => {
  it('shows each view its own count', async () => {
    const { counts } = await renderViewTabs();

    expect(counts.map((count) => count.textContent)).toEqual(['148', '7']);
  });

  it('keeps the count readable against the shown tab itself', async () => {
    const { counts, tabs } = await renderViewTabs();
    const shownIndex = tabs.findIndex((tab) => tab.getAttribute('aria-selected') === 'true');
    const count = counts[shownIndex]!;
    const style = getComputedStyle(count);

    // The shown fill is translucent, so measure against what it composites to
    // over the surface beneath.
    const effectiveTabBg = compositeColors(
      getComputedStyle(tabs[shownIndex]!).backgroundColor,
      getComputedStyle(document.body).backgroundColor
    );
    const ratio = getContrastRatio(style.color, effectiveTabBg, Number(style.opacity));

    expect(ratio).toBeGreaterThanOrEqual(4.5);
  });
});
