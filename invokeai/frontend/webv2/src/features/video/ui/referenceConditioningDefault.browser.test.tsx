/* oxlint-disable react-perf/jsx-no-new-function-as-prop */
import type { GalleryVideoItem } from '@features/gallery';
import type * as GalleryModule from '@features/gallery';
import type { VideoReferenceItem, VideoWidgetValues } from '@features/video/core/types';

import { ChakraProvider } from '@chakra-ui/react';
import { DndContext } from '@dnd-kit/core';
import { VideoUiProvider } from '@features/video/ui/VideoUiContext';
import { system } from '@theme/system';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { VideoReferenceListField } from './VideoReferenceListField';

/**
 * Verify gallery mediaOrigin reaches conditioning defaults so wrapped audio cannot condition on its waveform
 * image.
 */

const galleryMocks = vi.hoisted(() => ({ resolve: vi.fn(), uploadVideo: vi.fn() }));

vi.mock('@features/gallery', async (importOriginal) => {
  const actual = await importOriginal<typeof GalleryModule>();

  return {
    ...actual,
    galleryItems: { ...actual.galleryItems, resolve: galleryMocks.resolve },
    galleryTransfers: { ...actual.galleryTransfers, uploadVideo: galleryMocks.uploadVideo },
  };
});

const i18n = createInstance();

void i18n.use(initReactI18next).init({ fallbackLng: 'en', initAsync: false, lng: 'en', resources: { en: {} } });

const videoItem = (mediaOrigin?: string): GalleryVideoItem => ({
  boardId: 'none',
  category: 'user',
  createdAt: '2026-09-01T12:00:00Z',
  durationSeconds: 4,
  fps: 24,
  fullUrl: '/videos/clip.mp4',
  height: 360,
  isIntermediate: false,
  kind: 'video',
  ...(mediaOrigin ? { mediaOrigin } : {}),
  name: 'clip.mp4',
  starred: false,
  thumbnailUrl: '/thumbnails/clip.webp',
  width: 640,
});

const adapter = {
  findInGallery: () => undefined,
  getUploadBoardId: () => 'none',
  patchValues: (_values: Partial<VideoWidgetValues>) => undefined,
  playVideoSpanInPreview: () => null,
  projectId: 'project-1',
  rawValues: {},
  reportError: () => undefined,
  showPromptSyntaxHighlighting: false,
  touchGalleryImages: () => undefined,
  videoSpanPlayback: { getState: () => null, subscribe: () => () => undefined },
};

const NO_REFERENCES: VideoReferenceItem[] = [];

let container: HTMLDivElement;
let root: Root;

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

beforeEach(() => {
  galleryMocks.resolve.mockReset();
  galleryMocks.uploadVideo.mockReset();
  container = document.createElement('div');
  document.body.append(container);
  root = createRoot(container);
});

afterEach(() => {
  act(() => root.unmount());
  container.remove();
});

/** Uploads one video file through the list's own picker and returns the reference it added. */
const addUploadedVideo = async (mediaOrigin?: string): Promise<VideoReferenceItem[]> => {
  galleryMocks.uploadVideo.mockResolvedValue({ name: 'clip.mp4' });
  galleryMocks.resolve.mockResolvedValue(videoItem(mediaOrigin));

  let references: VideoReferenceItem[] = [];
  const onChange = (update: (current: VideoReferenceItem[]) => VideoReferenceItem[]) => {
    references = update(references);
  };

  act(() => {
    root.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <VideoUiProvider adapter={adapter}>
            <DndContext>
              <VideoReferenceListField
                maxImages={9}
                maxVideos={3}
                references={NO_REFERENCES}
                targetArea={null}
                onChange={onChange}
              />
            </DndContext>
          </VideoUiProvider>
        </ChakraProvider>
      </I18nextProvider>
    );
  });

  const inputs = [...container.querySelectorAll<HTMLInputElement>('input[type="file"]')];
  const videoInput = inputs.find((input) => input.accept.includes('audio/*'));

  expect(videoInput).toBeDefined();

  const file = new File([new Uint8Array([0])], 'song.mp3', { type: 'audio/mpeg' });
  const transfer = new DataTransfer();

  transfer.items.add(file);

  await act(async () => {
    Object.defineProperty(videoInput!, 'files', { configurable: true, value: transfer.files });
    videoInput!.dispatchEvent(new Event('change', { bubbles: true }));
    await Promise.resolve();
  });

  return references;
};

describe('the conditioning an added video reference starts on', () => {
  it('starts a wrapped audio upload on its soundtrack alone', async () => {
    const references = await addUploadedVideo('audio_upload');

    expect(references).toHaveLength(1);
    expect(references[0]).toMatchObject({ conditioning: 'audio', kind: 'video' });
  });

  it('starts an ordinary video on video + audio', async () => {
    const references = await addUploadedVideo();

    expect(references).toHaveLength(1);
    expect(references[0]).toMatchObject({ conditioning: 'video_audio', kind: 'video' });
  });

  it('resolves the item once, with no second metadata request to fall back from', async () => {
    await addUploadedVideo('audio_upload');

    expect(galleryMocks.resolve).toHaveBeenCalledTimes(1);
  });
});
