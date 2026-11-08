import type { GalleryItem } from '@features/gallery';

import { ChakraProvider, Flex } from '@chakra-ui/react';
import { DndContext } from '@dnd-kit/core';
import { system } from '@theme/system';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { expect, it, vi } from 'vitest';
import { page } from 'vitest/browser';

import type { PreviewSwipeNavigation } from './usePreviewSwipe';

import { PreviewFrame } from './PreviewFrame';
import { PREVIEW_NAV_STAGE_CSS, PreviewNavButtons } from './PreviewNavButtons';

vi.mock('@features/identity', () => ({
  getAuthSession: () => ({ accountEpoch: 1 }),
  refreshProtectedMediaCookie: () => Promise.resolve(false),
}));

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const i18n = createInstance();
await i18n.use(initReactI18next).init({
  interpolation: { escapeValue: false },
  lng: 'en',
  resources: { en: { translation: await fetch('/locales/en.json').then((response) => response.json()) } },
});
// Preview loads the player with its first video; loaded up front, it mounts in the render that asks for it.
await import('@platform/ui/video-player/VideoPlayer');

const neighbor: GalleryItem = {
  boardId: 'none',
  category: 'general',
  createdAt: '2026-01-01T00:00:00Z',
  fullUrl: '/neighbor.png',
  height: 64,
  isIntermediate: false,
  kind: 'image',
  name: 'neighbor',
  starred: false,
  thumbnailUrl: '/neighbor.png',
  width: 64,
};
const navigation: PreviewSwipeNavigation = {
  neighbors: { next: { item: neighbor, kind: 'item' }, previous: { item: neighbor, kind: 'item' } },
  onNavigate: () => Promise.resolve(true),
};

const SOURCE = {
  fps: 10,
  itemKey: 'video:clip',
  kind: 'video',
  label: 'Video clip',
  poster: '',
  src: '/scripts/mock-assets/fixture-video.mp4',
} as const;

const overlaps = (a: DOMRect, b: DOMRect): boolean =>
  a.left < b.right && a.right > b.left && a.top < b.bottom && a.bottom > b.top;

// A 16:9 clip in stages down to a small floating window, where the frame fills the stage and its bar is most of it.
it.each([
  [480, 300],
  [290, 200],
  [240, 160],
  [220, 140],
  [200, 120],
])("keeps a video's arrows off its control bar in a %i x %i stage", async (width, height) => {
  const host = document.body.appendChild(document.createElement('div'));
  const root = createRoot(host);

  try {
    await act(() => {
      root.render(
        <I18nextProvider i18n={i18n}>
          <ChakraProvider value={system}>
            <DndContext>
              <Flex
                css={PREVIEW_NAV_STAGE_CSS}
                direction="column"
                h={`${height}px`}
                position="relative"
                w={`${width}px`}
              >
                <PreviewFrame
                  frameHeight={1080}
                  frameWidth={1920}
                  isLive={false}
                  padding="3"
                  shouldAntialiasLiveImage
                  source={SOURCE}
                  swipe={navigation}
                  variant="framed"
                />
                <PreviewNavButtons isVideo navigation={navigation} />
              </Flex>
            </DndContext>
          </ChakraProvider>
        </I18nextProvider>
      );
    });
    await expect.element(page.getByRole('button', { name: 'Loop' })).toBeEnabled();

    const arrows = [...host.querySelectorAll('[data-preview-nav] button')].map((arrow) =>
      arrow.getBoundingClientRect()
    );
    const controls = [
      ...host.querySelectorAll('[data-video-player-controls] button, [data-video-player-controls] [role="slider"]'),
    ]
      .filter((control) => control.checkVisibility())
      .map((control) => ({ box: control.getBoundingClientRect(), name: control.getAttribute('aria-label') }));

    expect(arrows).toHaveLength(2);
    expect(controls.length).toBeGreaterThan(0);
    expect(
      controls.filter((control) => arrows.some((arrow) => overlaps(arrow, control.box))).map((control) => control.name)
    ).toEqual([]);
  } finally {
    await act(() => root.unmount());
    host.remove();
  }
});
