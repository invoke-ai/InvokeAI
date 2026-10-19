/* oxlint-disable react-perf/jsx-no-new-function-as-prop */
import type * as GalleryModule from '@features/gallery';
import type { VideoReferenceItem, VideoWidgetValues } from '@features/video/core/types';

import { ChakraProvider } from '@chakra-ui/react';
import { DndContext } from '@dnd-kit/core';
import { VideoUiProvider } from '@features/video/ui/VideoUiContext';
import { ApiError } from '@platform/transport/http';
import { system } from '@theme/system';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { VideoReferenceListField } from './VideoReferenceListField';

/**
 * An upload the board refused leaves the listing the board was chosen from stale: that listing is read again at once
 * so an upload that follows the read is routed by it. Other failures leave the listing alone.
 */

const galleryMocks = vi.hoisted(() => ({ upload: vi.fn() }));

vi.mock('@features/gallery', async (importOriginal) => {
  const actual = await importOriginal<typeof GalleryModule>();

  return { ...actual, galleryTransfers: { ...actual.galleryTransfers, upload: galleryMocks.upload } };
});

const i18n = createInstance();

void i18n.use(initReactI18next).init({ fallbackLng: 'en', initAsync: false, lng: 'en', resources: { en: {} } });

const getUploadBoardId = vi.fn(() => Promise.resolve('board-1'));
const refreshBoardDestinations = vi.fn();
const reportError = vi.fn();
const touchGalleryImages = vi.fn();
const adapter = {
  findInGallery: () => undefined,
  getUploadBoardId,
  patchValues: (_values: Partial<VideoWidgetValues>) => undefined,
  playVideoSpanInPreview: () => null,
  projectId: 'project-1',
  rawValues: {},
  readValues: () => ({}),
  refreshBoardDestinations,
  reportError,
  showPromptSyntaxHighlighting: false,
  touchGalleryImages,
  videoSpanPlayback: { getState: () => null, subscribe: () => () => undefined },
};

const NO_REFERENCES: VideoReferenceItem[] = [];

let container: HTMLDivElement;
let root: Root;

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

beforeEach(() => {
  vi.clearAllMocks();
  container = document.createElement('div');
  document.body.append(container);
  root = createRoot(container);
});

afterEach(() => {
  act(() => root.unmount());
  container.remove();
});

/** Uploads one image file through the list's own picker. */
const uploadImage = async (): Promise<void> => {
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
                onChange={() => undefined}
              />
            </DndContext>
          </VideoUiProvider>
        </ChakraProvider>
      </I18nextProvider>
    );
  });

  const imageInput = [...container.querySelectorAll<HTMLInputElement>('input[type="file"]')].find((input) =>
    input.accept.includes('image/png')
  );

  expect(imageInput).toBeDefined();

  const transfer = new DataTransfer();

  transfer.items.add(new File([new Uint8Array([0])], 'still.png', { type: 'image/png' }));

  await act(async () => {
    Object.defineProperty(imageInput!, 'files', { configurable: true, value: transfer.files });
    imageInput!.dispatchEvent(new Event('change', { bubbles: true }));
    await vi.waitFor(() => expect(reportError).toHaveBeenCalledTimes(1));
  });
};

describe('an upload the board refuses', () => {
  it.each([
    [
      'refuses the board (403)',
      new ApiError(JSON.stringify({ detail: 'Not authorized to change this board' }), 403),
      1,
    ],
    ['no longer has the board (404)', new ApiError(JSON.stringify({ detail: 'Board not found' }), 404), 1],
    ['fails otherwise', new Error('upload failed'), 0],
  ])('reads the board listing again only when the server %s', async (_case, error, refreshes) => {
    galleryMocks.upload.mockRejectedValue(error);

    await uploadImage();

    expect(galleryMocks.upload).toHaveBeenCalledWith(expect.any(File), 'board-1', expect.anything());
    expect(refreshBoardDestinations).toHaveBeenCalledTimes(refreshes);
    expect(touchGalleryImages).not.toHaveBeenCalled();
    expect(reportError).toHaveBeenCalledWith(error.message);
  });
});
