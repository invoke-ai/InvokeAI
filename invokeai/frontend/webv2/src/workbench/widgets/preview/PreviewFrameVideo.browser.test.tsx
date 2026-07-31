/* oxlint-disable react-perf/jsx-no-new-object-as-prop */
import type { GalleryItemKey } from '@features/gallery';

import { ChakraProvider } from '@chakra-ui/react';
import { DndContext, PointerSensor, useDndMonitor, useSensor, useSensors, type DragStartEvent } from '@dnd-kit/core';
import { system } from '@theme/system';
import { createInstance } from 'i18next';
import { act, createRef, type Ref } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import {
  PreviewFrame,
  type PreviewMediaSource,
  type PreviewVideoFrameController,
  type PreviewVideoFrameCopyResult,
} from './PreviewFrame';

const identityMocks = vi.hoisted(() => ({
  accountEpoch: 7,
  refreshProtectedMediaCookie: vi.fn<() => Promise<boolean>>(),
}));

vi.mock('@features/identity', () => ({
  getAuthSession: () => ({ accountEpoch: identityMocks.accountEpoch }),
  refreshProtectedMediaCookie: identityMocks.refreshProtectedMediaCookie,
}));

const i18n = createInstance();
void i18n.use(initReactI18next).init({
  fallbackLng: 'en',
  initAsync: false,
  lng: 'en',
  resources: {
    en: {
      translation: {
        widgets: {
          preview: {
            dropToCompare: 'Drop to compare',
            resetZoom: 'Reset zoom',
            videoFailed: 'Video could not be loaded',
            videoRetry: 'Retry',
          },
        },
      },
    },
  },
});

type VideoSource = Extract<PreviewMediaSource, { kind: 'video' }>;

const videoSource: VideoSource = {
  itemKey: 'video:clip.mp4' as const,
  kind: 'video' as const,
  label: 'Video clip.mp4',
  poster: 'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" width="16" height="9"/>',
  src: 'data:audio/wav;base64,UklGRiUAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQEAAACA',
};
const invalidVideoSource: VideoSource = { ...videoSource, src: 'data:video/mp4;base64,AAAA' };

let host: HTMLDivElement | null = null;
let root: Root | null = null;
let onDragStart = vi.fn();
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const interact = (action: () => void, delay = 0): Promise<void> =>
  act(async () => {
    action();
    await new Promise<void>((resolve) => {
      globalThis.setTimeout(resolve, delay);
    });
  });

const pointer = (type: string, target: EventTarget, clientX: number, clientY: number): void => {
  target.dispatchEvent(
    new PointerEvent(type, { bubbles: true, button: 0, clientX, clientY, isPrimary: true, pointerId: 1 })
  );
};

const DragMonitor = () => {
  useDndMonitor({
    onDragStart: (event: DragStartEvent) => onDragStart(event.active.data.current),
  });
  return null;
};

const VideoHarness = ({
  isItemCurrent = () => true,
  onCopyAvailabilityChange,
  source = videoSource,
  videoControllerRef,
}: {
  isItemCurrent?: (itemKey: GalleryItemKey) => boolean;
  onCopyAvailabilityChange?: (itemKey: GalleryItemKey, isAvailable: boolean) => void;
  source?: VideoSource;
  videoControllerRef?: Ref<PreviewVideoFrameController>;
}) => {
  const sensors = useSensors(useSensor(PointerSensor, { activationConstraint: { distance: 6 } }));

  return (
    <DndContext sensors={sensors}>
      <DragMonitor />
      <div style={{ display: 'flex', height: 260, width: 420 }}>
        <PreviewFrame
          frameHeight={1080}
          frameWidth={1920}
          isLive={false}
          liveBadgeLabel="Generating"
          shouldAntialiasLiveImage
          source={source}
          variant="framed"
          isItemCurrent={isItemCurrent}
          onVideoCopyAvailabilityChange={onCopyAvailabilityChange}
          videoControllerRef={videoControllerRef}
        />
      </div>
    </DndContext>
  );
};

beforeEach(() => {
  identityMocks.accountEpoch = 7;
  identityMocks.refreshProtectedMediaCookie.mockReset().mockResolvedValue(true);
  onDragStart = vi.fn();
  host = document.createElement('div');
  host.style.cssText = 'height:320px;left:20px;position:fixed;top:20px;width:480px;';
  document.body.append(host);
  root = createRoot(host);
});

afterEach(async () => {
  await interact(() => root?.unmount());
  vi.restoreAllMocks();
  host?.remove();
  host = null;
  root = null;
});

describe('PreviewFrame native video arm', () => {
  it('renders the native player with the protected media contract attributes', async () => {
    await interact(() => {
      root?.render(
        <I18nextProvider i18n={i18n}>
          <ChakraProvider value={system}>
            <VideoHarness />
          </ChakraProvider>
        </I18nextProvider>
      );
    });

    const video = host?.querySelector<HTMLVideoElement>('video');

    expect(video).not.toBeNull();
    expect(host?.querySelectorAll('video')).toHaveLength(1);
    expect(video?.controls).toBe(true);
    expect(video?.playsInline).toBe(true);
    expect(video?.getAttribute('preload')).toBe('metadata');
    expect(video?.getAttribute('poster')).toBe(videoSource.poster);
    expect(video?.getAttribute('src')).toBe(videoSource.src);
    expect(video?.getAttribute('aria-label')).toBe(videoSource.label);
    expect(host?.querySelector('img[alt="Video clip.mp4"]')).toBeNull();
  });

  it('does not arm a drag or cancel wheel events from the video surface and native controls', async () => {
    await interact(() => {
      root?.render(
        <I18nextProvider i18n={i18n}>
          <ChakraProvider value={system}>
            <VideoHarness />
          </ChakraProvider>
        </I18nextProvider>
      );
    });

    const video = host!.querySelector<HTMLVideoElement>('video')!;
    const content = video.parentElement;

    expect(content).not.toBeNull();
    expect(content ? getComputedStyle(content).touchAction : '').not.toBe('none');

    const wheel = new WheelEvent('wheel', { bubbles: true, cancelable: true, deltaY: 100 });
    video.dispatchEvent(wheel);
    expect(wheel.defaultPrevented).toBe(false);

    await interact(() => pointer('pointerdown', video, 220, 230), 20);
    await interact(() => pointer('pointermove', video.ownerDocument, 260, 230), 50);
    await interact(() => pointer('pointerup', video.ownerDocument, 260, 230), 300);

    expect(onDragStart).not.toHaveBeenCalled();
    expect(document.body.textContent).not.toContain('Drop to compare');
  });

  it('lets a trusted browser media error enter protected-media recovery', async () => {
    const trustedErrors: Event[] = [];
    const recordTrustedMediaError = (event: Event): void => {
      if (event.isTrusted && event.target instanceof HTMLMediaElement) {
        trustedErrors.push(event);
      }
    };
    document.addEventListener('error', recordTrustedMediaError, true);
    identityMocks.refreshProtectedMediaCookie.mockResolvedValueOnce(false);

    try {
      await renderVideo(() => true, invalidVideoSource);
      await act(async () => {
        await vi.waitFor(
          () => {
            expect(trustedErrors.length).toBeGreaterThan(0);
            expect(identityMocks.refreshProtectedMediaCookie).toHaveBeenCalledOnce();
          },
          { timeout: 5_000 }
        );
      });
    } finally {
      document.removeEventListener('error', recordTrustedMediaError, true);
    }
  });

  it('refreshes the protected-media cookie once and reloads after the first media failure', async () => {
    const refresh = deferred<boolean>();
    identityMocks.refreshProtectedMediaCookie.mockReturnValueOnce(refresh.promise);
    await renderVideo();
    const video = getVideo();
    const load = vi.spyOn(video, 'load').mockImplementation(() => undefined);

    await interact(() => video.dispatchEvent(new Event('error')));

    expect(identityMocks.refreshProtectedMediaCookie).toHaveBeenCalledTimes(1);
    expect(load).not.toHaveBeenCalled();

    await interact(() => refresh.resolve(true));

    expect(load).toHaveBeenCalledTimes(1);
    expect(host?.textContent).not.toContain('Video could not be loaded');
  });

  it('does not reload or publish failure after the account changes while refresh is pending', async () => {
    const refresh = deferred<boolean>();
    identityMocks.refreshProtectedMediaCookie.mockReturnValueOnce(refresh.promise);
    await renderVideo();
    const video = getVideo();
    const load = vi.spyOn(video, 'load').mockImplementation(() => undefined);

    await interact(() => video.dispatchEvent(new Event('error')));
    expect(identityMocks.refreshProtectedMediaCookie).toHaveBeenCalledTimes(1);
    identityMocks.accountEpoch = 8;
    await interact(() => refresh.resolve(true));

    expect(load).not.toHaveBeenCalled();
    expect(host?.textContent).not.toContain('Video could not be loaded');
  });

  it('does not reload or publish failure after a different item becomes current', async () => {
    let isCurrent = true;
    const refresh = deferred<boolean>();
    identityMocks.refreshProtectedMediaCookie.mockReturnValueOnce(refresh.promise);
    await renderVideo(() => isCurrent);
    const video = getVideo();
    const load = vi.spyOn(video, 'load').mockImplementation(() => undefined);

    await interact(() => video.dispatchEvent(new Event('error')));
    expect(identityMocks.refreshProtectedMediaCookie).toHaveBeenCalledTimes(1);
    isCurrent = false;
    await interact(() => refresh.resolve(true));

    expect(load).not.toHaveBeenCalled();
    expect(host?.textContent).not.toContain('Video could not be loaded');
  });

  it('shows a poster-backed failure state when refresh fails or the automatic reload fails', async () => {
    identityMocks.refreshProtectedMediaCookie.mockResolvedValueOnce(false);
    await renderVideo();
    const firstVideo = getVideo();
    vi.spyOn(firstVideo, 'load').mockImplementation(() => undefined);

    await interact(() => firstVideo.dispatchEvent(new Event('error')));

    expect(host?.textContent).toContain('Video could not be loaded');
    expect(host?.querySelector<HTMLButtonElement>('button')?.textContent).toBe('Retry');
    expect(
      [...(host?.querySelectorAll<HTMLImageElement>('img') ?? [])].some(
        (image) => image.getAttribute('src') === videoSource.poster
      )
    ).toBe(true);

    identityMocks.refreshProtectedMediaCookie.mockResolvedValueOnce(true);
    await renderVideo(() => true, { ...videoSource, itemKey: 'video:second.mp4' });
    const secondVideo = getVideo();
    const load = vi.spyOn(secondVideo, 'load').mockImplementation(() => undefined);

    await interact(() => secondVideo.dispatchEvent(new Event('error')));
    expect(load).toHaveBeenCalledTimes(1);
    await interact(() => secondVideo.dispatchEvent(new Event('error')));

    expect(identityMocks.refreshProtectedMediaCookie).toHaveBeenCalledTimes(2);
    expect(host?.textContent).toContain('Video could not be loaded');
  });

  it('gives a manual retry a fresh automatic refresh budget', async () => {
    await renderVideo();
    const video = getVideo();
    const load = vi.spyOn(video, 'load').mockImplementation(() => undefined);

    await interact(() => video.dispatchEvent(new Event('error')));
    await interact(() => video.dispatchEvent(new Event('error')));
    expect(identityMocks.refreshProtectedMediaCookie).toHaveBeenCalledTimes(1);

    await interact(() => host?.querySelector<HTMLButtonElement>('button')?.click());
    expect(load).toHaveBeenCalledTimes(2);
    expect(host?.textContent).not.toContain('Video could not be loaded');

    const secondRefresh = deferred<boolean>();
    identityMocks.refreshProtectedMediaCookie.mockReturnValueOnce(secondRefresh.promise);
    await interact(() => video.dispatchEvent(new Event('error')));
    expect(identityMocks.refreshProtectedMediaCookie).toHaveBeenCalledTimes(2);
    await interact(() => secondRefresh.resolve(true));

    expect(load).toHaveBeenCalledTimes(3);
  });

  it('keys retry state to the selected video item', async () => {
    await renderVideo();
    const firstVideo = getVideo();
    vi.spyOn(firstVideo, 'load').mockImplementation(() => undefined);

    await interact(() => firstVideo.dispatchEvent(new Event('error')));
    await interact(() => firstVideo.dispatchEvent(new Event('error')));
    expect(host?.textContent).toContain('Video could not be loaded');

    const nextSource = {
      ...videoSource,
      itemKey: 'video:next.mp4' as const,
      label: 'Video next.mp4',
    };
    await renderVideo(() => true, nextSource);
    const nextVideo = getVideo();
    vi.spyOn(nextVideo, 'load').mockImplementation(() => undefined);

    expect(host?.textContent).not.toContain('Video could not be loaded');
    await interact(() => nextVideo.dispatchEvent(new Event('error')));

    expect(identityMocks.refreshProtectedMediaCookie).toHaveBeenCalledTimes(2);
  });

  it('enables frame copy only for current intrinsic frame data and ClipboardItem write support', async () => {
    setClipboardUnsupported();
    const controllerRef = createRef<PreviewVideoFrameController>();
    const onCopyAvailabilityChange = vi.fn();
    await renderVideo(() => true, videoSource, controllerRef, onCopyAvailabilityChange);
    const video = getVideo();

    setVideoFrameState(video, { height: 1080, readyState: HTMLMediaElement.HAVE_CURRENT_DATA, width: 1920 });
    await interact(() => video.dispatchEvent(new Event('loadeddata')));

    expect(controllerRef.current?.isCopyAvailable()).toBe(false);
    expect(onCopyAvailabilityChange).toHaveBeenLastCalledWith(videoSource.itemKey, false);

    installClipboardSupport();
    setVideoFrameState(video, { height: 0, readyState: HTMLMediaElement.HAVE_CURRENT_DATA, width: 1920 });
    await interact(() => video.dispatchEvent(new Event('resize')));
    expect(controllerRef.current?.isCopyAvailable()).toBe(false);

    setVideoFrameState(video, { height: 1080, readyState: HTMLMediaElement.HAVE_METADATA, width: 1920 });
    await interact(() => video.dispatchEvent(new Event('waiting')));
    expect(controllerRef.current?.isCopyAvailable()).toBe(false);

    setVideoFrameState(video, { height: 1080, readyState: HTMLMediaElement.HAVE_CURRENT_DATA, width: 1920 });
    await interact(() => video.dispatchEvent(new Event('canplay')));

    expect(controllerRef.current?.isCopyAvailable()).toBe(true);
    expect(onCopyAvailabilityChange).toHaveBeenLastCalledWith(videoSource.itemKey, true);
  });

  it('draws the intrinsic video frame, encodes PNG, and writes one ClipboardItem', async () => {
    const controllerRef = createRef<PreviewVideoFrameController>();
    const { clipboardItems, write } = installClipboardSupport();
    const drawImage = vi.fn();
    const canvases: HTMLCanvasElement[] = [];
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockImplementation(function (this: HTMLCanvasElement) {
      canvases.push(this);
      return { drawImage } as unknown as CanvasRenderingContext2D;
    });
    const png = new Blob(['frame'], { type: 'image/png' });
    vi.spyOn(HTMLCanvasElement.prototype, 'toBlob').mockImplementation((callback, type) => {
      expect(type).toBe('image/png');
      callback(png);
    });
    await renderVideo(() => true, videoSource, controllerRef);
    const video = getVideo();
    setVideoFrameState(video, { height: 720, readyState: HTMLMediaElement.HAVE_CURRENT_DATA, width: 1280 });

    const result = await copyFrame(controllerRef);

    expect(result).toEqual({ ok: true });
    expect(canvases[0]?.width).toBe(1280);
    expect(canvases[0]?.height).toBe(720);
    expect(drawImage).toHaveBeenCalledWith(video, 0, 0, 1280, 720);
    expect(clipboardItems).toHaveLength(1);
    expect(clipboardItems[0]?.['image/png']).toBe(png);
    expect(write).toHaveBeenCalledOnce();
  });

  it('distinguishes unsupported Clipboard APIs from media without a current frame', async () => {
    const controllerRef = createRef<PreviewVideoFrameController>();
    setClipboardUnsupported();
    await renderVideo(() => true, videoSource, controllerRef);
    const video = getVideo();
    setVideoFrameState(video, { height: 1080, readyState: HTMLMediaElement.HAVE_CURRENT_DATA, width: 1920 });

    await expect(copyFrame(controllerRef)).resolves.toEqual({ ok: false, reason: 'unsupported' });

    installClipboardSupport();
    setVideoFrameState(video, { height: 1080, readyState: HTMLMediaElement.HAVE_METADATA, width: 1920 });

    await expect(copyFrame(controllerRef)).resolves.toEqual({ ok: false, reason: 'not-ready' });
  });

  it('reports canvas draw and taint failures without touching the clipboard', async () => {
    const controllerRef = createRef<PreviewVideoFrameController>();
    const { write } = installClipboardSupport();
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({
      drawImage: () => {
        throw new DOMException('The canvas is tainted.', 'SecurityError');
      },
    } as unknown as CanvasRenderingContext2D);
    await renderVideo(() => true, videoSource, controllerRef);
    setVideoFrameState(getVideo(), { height: 1080, readyState: HTMLMediaElement.HAVE_CURRENT_DATA, width: 1920 });

    await expect(copyFrame(controllerRef)).resolves.toEqual({ ok: false, reason: 'draw-failed' });
    expect(write).not.toHaveBeenCalled();

    vi.mocked(HTMLCanvasElement.prototype.getContext).mockReturnValue({
      drawImage: vi.fn(),
    } as unknown as CanvasRenderingContext2D);
    vi.spyOn(HTMLCanvasElement.prototype, 'toBlob').mockImplementation(() => {
      throw new DOMException('The canvas is tainted.', 'SecurityError');
    });

    await expect(copyFrame(controllerRef)).resolves.toEqual({ ok: false, reason: 'draw-failed' });
    expect(write).not.toHaveBeenCalled();
  });

  it.each([
    {
      configure: () => vi.spyOn(HTMLCanvasElement.prototype, 'toBlob').mockImplementation((callback) => callback(null)),
      label: 'null PNG encoding',
    },
    {
      configure: () =>
        vi.spyOn(HTMLCanvasElement.prototype, 'toBlob').mockImplementation(() => {
          throw new Error('encoder rejected');
        }),
      label: 'rejected PNG encoding',
    },
  ])('reports $label without writing', async ({ configure }) => {
    const controllerRef = createRef<PreviewVideoFrameController>();
    const { write } = installClipboardSupport();
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({
      drawImage: vi.fn(),
    } as unknown as CanvasRenderingContext2D);
    configure();
    await renderVideo(() => true, videoSource, controllerRef);
    setVideoFrameState(getVideo(), { height: 1080, readyState: HTMLMediaElement.HAVE_CURRENT_DATA, width: 1920 });

    await expect(copyFrame(controllerRef)).resolves.toEqual({ ok: false, reason: 'encode-failed' });
    expect(write).not.toHaveBeenCalled();
  });

  it('reports ClipboardItem construction and clipboard write rejection', async () => {
    const controllerRef = createRef<PreviewVideoFrameController>();
    const { write } = installClipboardSupport();
    installSuccessfulCanvas();
    await renderVideo(() => true, videoSource, controllerRef);
    setVideoFrameState(getVideo(), { height: 1080, readyState: HTMLMediaElement.HAVE_CURRENT_DATA, width: 1920 });
    Object.defineProperty(globalThis, 'ClipboardItem', {
      configurable: true,
      value: class {
        constructor() {
          throw new Error('ClipboardItem rejected');
        }
      },
    });

    await expect(copyFrame(controllerRef)).resolves.toEqual({ ok: false, reason: 'clipboard-failed' });

    installClipboardSupport().write.mockRejectedValueOnce(new Error('permission denied'));
    await expect(copyFrame(controllerRef)).resolves.toEqual({ ok: false, reason: 'clipboard-failed' });
    expect(write).not.toHaveBeenCalled();
  });

  it('does not report success after the account epoch changes during clipboard write', async () => {
    const controllerRef = createRef<PreviewVideoFrameController>();
    const writeResult = deferred<void>();
    const { write } = installClipboardSupport();
    write.mockReturnValueOnce(writeResult.promise);
    installSuccessfulCanvas();
    await renderVideo(() => true, videoSource, controllerRef);
    setVideoFrameState(getVideo(), { height: 1080, readyState: HTMLMediaElement.HAVE_CURRENT_DATA, width: 1920 });

    const result = copyFrame(controllerRef);
    await interact(() => undefined);
    identityMocks.accountEpoch = 8;
    await interact(() => writeResult.resolve());

    await expect(result).resolves.toEqual({ ok: false, reason: 'stale' });
  });

  it('does not report success after a different GalleryItemKey becomes current', async () => {
    let isCurrent = true;
    const controllerRef = createRef<PreviewVideoFrameController>();
    const writeResult = deferred<void>();
    const { write } = installClipboardSupport();
    write.mockReturnValueOnce(writeResult.promise);
    installSuccessfulCanvas();
    await renderVideo(() => isCurrent, videoSource, controllerRef);
    setVideoFrameState(getVideo(), { height: 1080, readyState: HTMLMediaElement.HAVE_CURRENT_DATA, width: 1920 });

    const result = copyFrame(controllerRef);
    await interact(() => undefined);
    isCurrent = false;
    await interact(() => writeResult.resolve());

    await expect(result).resolves.toEqual({ ok: false, reason: 'stale' });
  });
});

const renderVideo = async (
  isItemCurrent?: (itemKey: GalleryItemKey) => boolean,
  source: VideoSource = videoSource,
  videoControllerRef?: Ref<PreviewVideoFrameController>,
  onCopyAvailabilityChange?: (itemKey: GalleryItemKey, isAvailable: boolean) => void
): Promise<void> => {
  await interact(() => {
    root?.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <VideoHarness
            isItemCurrent={isItemCurrent}
            onCopyAvailabilityChange={onCopyAvailabilityChange}
            source={source}
            videoControllerRef={videoControllerRef}
          />
        </ChakraProvider>
      </I18nextProvider>
    );
  });
};

const getVideo = (): HTMLVideoElement => {
  const video = host?.querySelector<HTMLVideoElement>('video');

  if (!video) {
    throw new Error('Expected the preview video to be mounted.');
  }

  return video;
};

const deferred = <T,>() => {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((resolvePromise) => {
    resolve = resolvePromise;
  });

  return { promise, resolve };
};

const originalClipboardDescriptor = Object.getOwnPropertyDescriptor(navigator, 'clipboard');
const originalClipboardItemDescriptor = Object.getOwnPropertyDescriptor(globalThis, 'ClipboardItem');

const restoreClipboardGlobals = (): void => {
  if (originalClipboardDescriptor) {
    Object.defineProperty(navigator, 'clipboard', originalClipboardDescriptor);
  } else {
    Reflect.deleteProperty(navigator, 'clipboard');
  }

  if (originalClipboardItemDescriptor) {
    Object.defineProperty(globalThis, 'ClipboardItem', originalClipboardItemDescriptor);
  } else {
    Reflect.deleteProperty(globalThis, 'ClipboardItem');
  }
};

afterEach(restoreClipboardGlobals);

const setClipboardUnsupported = (): void => {
  Object.defineProperty(navigator, 'clipboard', { configurable: true, value: undefined });
  Reflect.deleteProperty(globalThis, 'ClipboardItem');
};

const installClipboardSupport = () => {
  const clipboardItems: Record<string, Blob>[] = [];
  const write = vi.fn<(items: ClipboardItems) => Promise<void>>().mockResolvedValue(undefined);

  class TestClipboardItem {
    constructor(data: Record<string, Blob>) {
      clipboardItems.push(data);
    }
  }

  Object.defineProperty(globalThis, 'ClipboardItem', { configurable: true, value: TestClipboardItem });
  Object.defineProperty(navigator, 'clipboard', { configurable: true, value: { write } });

  return { clipboardItems, write };
};

const setVideoFrameState = (
  video: HTMLVideoElement,
  state: { height: number; readyState: number; width: number }
): void => {
  Object.defineProperties(video, {
    readyState: { configurable: true, value: state.readyState },
    videoHeight: { configurable: true, value: state.height },
    videoWidth: { configurable: true, value: state.width },
  });
};

const installSuccessfulCanvas = (): void => {
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({
    drawImage: vi.fn(),
  } as unknown as CanvasRenderingContext2D);
  vi.spyOn(HTMLCanvasElement.prototype, 'toBlob').mockImplementation((callback) => {
    callback(new Blob(['frame'], { type: 'image/png' }));
  });
};

const copyFrame = (controllerRef: {
  current: PreviewVideoFrameController | null;
}): Promise<PreviewVideoFrameCopyResult> => {
  const controller = controllerRef.current;

  expect(controller).not.toBeNull();

  if (!controller) {
    return Promise.reject(new Error('Expected a video frame controller.'));
  }

  return controller.copyCurrentFrame();
};
