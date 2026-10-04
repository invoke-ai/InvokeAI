import type { GalleryUiAdapter } from '@features/gallery/react';
import type { VideoSourceClip } from '@features/video/core/types';

import { ChakraProvider } from '@chakra-ui/react';
import { DndContext } from '@dnd-kit/core';
import { GalleryUiProvider } from '@features/gallery/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import i18next from 'i18next';
import { act, type ReactNode } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { VideoFrameImageField } from './VideoFrameImageField';
import { VideoSourceClipField } from './VideoSourceClipField';
import { VideoUiProvider, type VideoUiAdapter } from './VideoUiContext';

/** Conditioning slots must pass media kind with name; video names cannot resolve through the image endpoint. */
const i18n = i18next.createInstance();
await i18n.use(initReactI18next).init({
  fallbackLng: 'en',
  lng: 'en',
  resources: {
    en: {
      translation: {
        widgets: {
          gallery: {
            findNamedInGallery: 'Find {{name}} in Gallery',
            picker: {
              chooseImage: 'Choose image',
              dropHint: 'Drop',
              removeImage: 'Remove image',
              replaceHint: 'Replace',
              replaceImage: 'Replace image',
              upload: 'Upload',
            },
          },
          video: {
            playSelection: 'Play selection in Preview',
            trim: 'Trim',
            trimEnd: 'Trim end',
            trimEndShort: 'End',
            trimHelp: 'help',
            trimStart: 'Trim start',
            trimStartShort: 'Start',
          },
        },
      },
    },
  },
});

let host: HTMLDivElement;
let root: Root;
let queryClient: QueryClient;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const findInGallery = vi.fn();
const videoAdapter = {
  findInGallery,
  getUploadBoardId: () => 'none',
  patchValues: vi.fn(),
  playVideoSpanInPreview: vi.fn(),
  reportError: vi.fn(),
  touchGalleryImages: vi.fn(),
  videoSpanPlayback: { getState: () => null, subscribe: () => () => undefined },
} as unknown as VideoUiAdapter;

const galleryAdapter = {
  gallery: { selectBoard: vi.fn(), selectItem: vi.fn(), setView: vi.fn() },
  galleryValues: {},
  notifications: { add: vi.fn(), reportError: vi.fn() },
  widgets: { openGallery: () => true, patchGalleryValues: vi.fn() },
} as unknown as GalleryUiAdapter;

const FIRST_FRAME = { height: 512, image_name: 'first.png', width: 512 };
const CLIP: VideoSourceClip = {
  endFrame: 47,
  fps: 24,
  height: 480,
  numFrames: 48,
  startFrame: 0,
  video_name: 'source.mp4',
  width: 832,
};

const render = async (children: ReactNode): Promise<void> => {
  await act(() =>
    root.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <QueryClientProvider client={queryClient}>
            <GalleryUiProvider adapter={galleryAdapter}>
              <DndContext>
                <VideoUiProvider adapter={videoAdapter}>{children}</VideoUiProvider>
              </DndContext>
            </GalleryUiProvider>
          </QueryClientProvider>
        </ChakraProvider>
      </I18nextProvider>
    )
  );
};

const findButtons = (name: string): HTMLButtonElement[] => [
  ...host.querySelectorAll<HTMLButtonElement>(`button[aria-label="Find ${name} in Gallery"]`),
];

beforeEach(() => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  findInGallery.mockClear();
});

afterEach(async () => {
  await act(() => root.unmount());
  host.remove();
  queryClient.clear();
});

describe('video conditioning find-in-gallery badges', () => {
  it('reveals a keyframe image as an image', async () => {
    await render(
      <VideoFrameImageField dropId="first-frame" dropLabel="Drop First Frame" image={FIRST_FRAME} onChange={vi.fn()} />
    );

    await act(() => findButtons('first.png')[0]?.click());

    expect(findInGallery).toHaveBeenCalledExactlyOnceWith({ kind: 'image', name: 'first.png' });
  });

  it('reveals the initial video as a video, once, from its start bound', async () => {
    await render(<VideoSourceClipField sourceVideo={CLIP} onChange={vi.fn()} />);

    // Both bounds reference one clip; expose one find control.
    const bounds = findButtons('source.mp4');

    expect(bounds).toHaveLength(1);

    await act(() => bounds[0]?.click());

    expect(findInGallery).toHaveBeenCalledExactlyOnceWith({ kind: 'video', name: 'source.mp4' });
  });

  it('offers no badge on an empty keyframe slot', async () => {
    await render(
      <VideoFrameImageField dropId="last-frame" dropLabel="Drop Last Frame" image={null} onChange={vi.fn()} />
    );

    expect(host.querySelector('button[aria-label^="Find "]')).toBeNull();
  });
});
