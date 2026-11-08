import { ChakraProvider } from '@chakra-ui/react';
import { DndContext } from '@dnd-kit/core';
import { system } from '@theme/system';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { expect, it, vi } from 'vitest';

import { PreviewFrame } from './PreviewFrame';

// Holds the player module back, so the frame renders before it arrives, as on a first video after boot.
const playerModule = vi.hoisted(() => {
  let release = (): void => {};
  const loaded = new Promise<void>((resolve) => {
    release = resolve;
  });

  return { loaded, release: () => release() };
});

vi.mock('@platform/ui/video-player/VideoPlayer', async (importOriginal) => {
  await playerModule.loaded;

  return importOriginal();
});

vi.mock('@features/identity', () => ({
  getAuthSession: () => ({ accountEpoch: 1 }),
  refreshProtectedMediaCookie: () => Promise.resolve(false),
}));

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const i18n = createInstance();
void i18n.use(initReactI18next).init({ initAsync: false, lng: 'en', resources: { en: { translation: {} } } });

const POSTER = 'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7';

const SOURCE = {
  itemKey: 'video:clip.mp4',
  kind: 'video',
  label: 'Video clip.mp4',
  poster: POSTER,
  src: '/scripts/mock-assets/fixture-video.mp4',
} as const;

it('shows the poster while the player loads, then the player in its place', async () => {
  const host = document.body.appendChild(document.createElement('div'));
  const root = createRoot(host);

  try {
    await act(async () => {
      root.render(
        <I18nextProvider i18n={i18n}>
          <ChakraProvider value={system}>
            <DndContext>
              <div style={{ display: 'flex', height: 260, width: 420 }}>
                <PreviewFrame
                  frameHeight={1080}
                  frameWidth={1920}
                  isLive={false}
                  shouldAntialiasLiveImage
                  source={SOURCE}
                  variant="framed"
                />
              </div>
            </DndContext>
          </ChakraProvider>
        </I18nextProvider>
      );
      await new Promise((resolve) => {
        setTimeout(resolve, 50);
      });
    });

    expect(host.querySelector('video')).toBeNull();
    expect(host.querySelector(`img[src="${POSTER}"]`)?.getAttribute('aria-hidden')).toBe('true');

    playerModule.release();

    // The mocked module resolves through the test server, which a loaded machine can keep for well over a second;
    // this test is about what shows first and what replaces it, not how fast.
    await vi.waitFor(
      async () => {
        await act(async () => {});
        expect(host.querySelector('video')?.getAttribute('poster')).toBe(POSTER);
      },
      { timeout: 10_000 }
    );
    expect(host.querySelector(`img[src="${POSTER}"]`)).toBeNull();
  } finally {
    await act(() => root.unmount());
    host.remove();
  }
});
