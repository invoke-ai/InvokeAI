/* oxlint-disable react-perf/jsx-no-new-object-as-prop */
import type { GalleryItemKey } from '@features/gallery';

import { ChakraProvider } from '@chakra-ui/react';
import { DndContext, PointerSensor, useDndMonitor, useSensor, useSensors, type DragStartEvent } from '@dnd-kit/core';
import { system } from '@theme/system';
import { createInstance } from 'i18next';
import { act, Activity, createRef, type Ref, StrictMode } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

import {
  PreviewFrame,
  type PreviewMediaSource,
  type PreviewVideoFrameController,
  type PreviewVideoFrameCopyResult,
} from './PreviewFrame';
import {
  clearVideoSpanPlaybackState,
  consumeVideoSpanPlaybackRequest,
  getVideoSpanPlaybackRequest,
  getVideoSpanPlaybackState,
  requestVideoSpanPlayback,
} from './spanPlaybackRequest';

const identityMocks = vi.hoisted(() => ({
  accountEpoch: 7,
  refreshProtectedMediaCookie: vi.fn<() => Promise<boolean>>(),
}));

vi.mock('@features/identity', async (importOriginal) => ({
  ...(await importOriginal<object>()),
  getAuthSession: () => ({ accountEpoch: identityMocks.accountEpoch }),
  refreshProtectedMediaCookie: identityMocks.refreshProtectedMediaCookie,
}));

const i18n = createInstance();
await i18n.use(initReactI18next).init({
  fallbackLng: 'en',
  interpolation: { escapeValue: false },
  lng: 'en',
  resources: { en: { translation: await fetch('/locales/en.json').then((response) => response.json()) } },
});

// Preview loads the player with its first video. Loaded up front, it mounts in the render that asks for it, before the
// element can load media the stubbed tests below stand in for.
await import('@platform/ui/video-player/VideoPlayer');

const FIXTURE_VIDEO_URL = '/scripts/mock-assets/fixture-video.mp4';

type VideoSource = Extract<PreviewMediaSource, { kind: 'video' }>;

const videoSource: VideoSource = {
  itemKey: 'video:clip.mp4' as const,
  kind: 'video' as const,
  label: 'Video clip.mp4',
  poster: 'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" width="16" height="9"/>',
  src: 'data:audio/wav;base64,UklGRiUAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQEAAACA',
};
const invalidVideoSource: VideoSource = { ...videoSource, src: 'data:video/mp4;base64,AAAA' };
const fixtureVideoSource: VideoSource = {
  ...videoSource,
  fps: 10,
  itemKey: 'video:fixture.mp4',
  src: FIXTURE_VIDEO_URL,
};

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

const DragMonitor = () => {
  useDndMonitor({
    onDragStart: (event: DragStartEvent) => onDragStart(event.active.data.current),
  });
  return null;
};

const VideoHarness = ({
  isItemCurrent = () => true,
  onCopyAvailabilityChange,
  onVideoLoopChange,
  source = videoSource,
  videoControllerRef,
  videoLoop,
}: {
  isItemCurrent?: (itemKey: GalleryItemKey) => boolean;
  onCopyAvailabilityChange?: (itemKey: GalleryItemKey, isAvailable: boolean) => void;
  onVideoLoopChange?: (loop: boolean) => void;
  source?: VideoSource;
  videoControllerRef?: Ref<PreviewVideoFrameController>;
  videoLoop?: boolean;
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
          shouldAntialiasLiveImage
          source={source}
          variant="framed"
          isItemCurrent={isItemCurrent}
          onVideoCopyAvailabilityChange={onCopyAvailabilityChange}
          videoControllerRef={videoControllerRef}
          videoLoop={videoLoop}
          onVideoLoopChange={onVideoLoopChange}
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
  vi.useRealTimers();
  clearSpanRequest();
  host?.remove();
  host = null;
  root = null;
});

describe('PreviewFrame video', () => {
  it('renders the player with the protected media contract attributes', async () => {
    await renderVideo();

    const video = getVideo();

    expect(host?.querySelectorAll('video')).toHaveLength(1);
    expect(video.controls).toBe(false);
    expect(video.playsInline).toBe(true);
    expect(video.getAttribute('preload')).toBe('metadata');
    expect(video.getAttribute('poster')).toBe(videoSource.poster);
    expect(video.getAttribute('src')).toBe(videoSource.src);
    await expect.element(page.getByRole('group', { name: videoSource.label })).toContainElement(video);
    await expect.element(page.getByRole('button', { name: 'Loop' })).toHaveAttribute('aria-pressed', 'true');
    expect(host?.querySelector('img[alt="Video clip.mp4"]')).toBeNull();
  });

  it('plays and loops the span a request published before the player mounted asked for', async () => {
    requestVideoSpanPlayback({ endSeconds: 4, itemKey: videoSource.itemKey, startSeconds: 2 });
    await renderVideo();
    const video = getVideo();
    const playback = stubSpanPlayback(video);

    await interact(() => video.dispatchEvent(new Event('loadedmetadata')));

    expect(playback.getTime()).toBe(2);
    expect(playback.play).toHaveBeenCalledTimes(1);
    expect(getVideoSpanPlaybackRequest()).toBeNull();

    await interact(() => {
      playback.setTime(4.02);
      video.dispatchEvent(new Event('timeupdate'));
    });

    expect(playback.getTime()).toBe(2);
  });

  it('drops a request the user has long since moved on from', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true, toFake: ['Date'] });
    requestVideoSpanPlayback({ endSeconds: 4, itemKey: videoSource.itemKey, startSeconds: 2 });
    vi.setSystemTime(Date.now() + 60_000);

    await renderVideo();
    const video = getVideo();
    const playback = stubSpanPlayback(video);

    await interact(() => video.dispatchEvent(new Event('loadedmetadata')));

    expect(playback.play).not.toHaveBeenCalled();
    expect(getVideoSpanPlaybackRequest()).toBeNull();
    expect(playback.getTime()).toBeLessThan(0.001);
  });

  it('hands the playhead back for good once the user scrubs', async () => {
    requestVideoSpanPlayback({ endSeconds: 4, itemKey: videoSource.itemKey, startSeconds: 2 });
    await renderVideo();
    const video = getVideo();
    const playback = stubSpanPlayback(video);

    await interact(() => video.dispatchEvent(new Event('loadedmetadata')));
    expect(playback.getTime()).toBe(2);

    await interact(() => playback.scrubTo(8));
    await interact(() => {
      playback.setTime(9);
      video.dispatchEvent(new Event('timeupdate'));
    });

    expect(playback.getTime()).toBe(9);
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

  it('shows a poster-backed failure state when refresh fails or the automatic reload fails', async () => {
    identityMocks.refreshProtectedMediaCookie.mockResolvedValueOnce(false);
    await renderVideo();
    const firstVideo = getVideo();
    vi.spyOn(firstVideo, 'load').mockImplementation(() => undefined);

    await interact(() => firstVideo.dispatchEvent(new Event('error')));

    expect(host?.textContent).toContain('Video could not be loaded');
    await expect.element(page.getByRole('button', { name: 'Retry' })).toBeVisible();
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
});

describe('PreviewFrame video span playback through the player', () => {
  it('loops a requested span within the clip while looping is on', async () => {
    await renderHarness({ source: fixtureVideoSource });
    const video = await loadedVideo();
    const seeks = recordSeeks(video);
    const token = requestVideoSpanPlayback({ endSeconds: 0.5, itemKey: fixtureVideoSource.itemKey, startSeconds: 0.2 });

    await expect.poll(() => video.paused).toBe(false);
    await expect.poll(() => getScrubberRoot().hasAttribute('data-playback-window')).toBe(true);

    await expect.poll(() => seeks.filter(isSpanStart).length, { timeout: 3000 }).toBeGreaterThanOrEqual(2);
    expect(video.paused).toBe(false);
    expect(getVideoSpanPlaybackState()).toMatchObject({ isPlaying: true, token });
    video.pause();
  });
});

describe('PreviewFrame video keep-alive', () => {
  it('stops playback when the shell hides the widget instead of unmounting it', async () => {
    await renderKeptVideo('visible');
    const video = getVideo();
    const playback = stubSpanPlayback(video);

    await interact(() => void video.play());
    await renderKeptVideo('hidden');

    // The element survives — this is the keep-alive path, not an unmount — but `display: none` does not stop media, so
    // the player has to pause it itself, once.
    expect(getVideo()).toBe(video);
    expect(video.checkVisibility()).toBe(false);
    expect(playback.pauseSpy).toHaveBeenCalledOnce();
    expect(video.paused).toBe(true);
  });
});

const renderKeptVideo = async (
  mode: 'hidden' | 'visible',
  { source, strict = false }: { source?: VideoSource; strict?: boolean } = {}
): Promise<void> => {
  const app = (
    <I18nextProvider i18n={i18n}>
      <ChakraProvider value={system}>
        <Activity mode={mode}>
          <VideoHarness source={source} />
        </Activity>
      </ChakraProvider>
    </I18nextProvider>
  );

  await interact(() => {
    root?.render(strict ? <StrictMode>{app}</StrictMode> : app);
  });
  await waitForPlayer();
};

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
  await waitForPlayer();
};

const renderHarness = async (props: Parameters<typeof VideoHarness>[0]): Promise<void> => {
  await interact(() => {
    root?.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <VideoHarness {...props} />
        </ChakraProvider>
      </I18nextProvider>
    );
  });
  await waitForPlayer();
};

const waitForPlayer = async (): Promise<void> => {
  if (host?.querySelector('video')) {
    return;
  }

  await vi.waitFor(async () => {
    // Inside act, React holds the retry the loaded player schedules until act ends.
    await act(async () => {});
    expect(host?.querySelector('video')).not.toBeNull();
  });
};

const loadedVideo = async (): Promise<HTMLVideoElement> => {
  const video = getVideo();

  await expect.poll(() => video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA, { timeout: 5000 }).toBe(true);
  await expect.element(page.getByRole('button', { name: 'Loop' })).toBeEnabled();
  // A span request follows a press in the Video panel; without that activation the browser refuses audible playback.
  await userEvent.keyboard('{Shift}');

  return video;
};

const recordSeeks = (video: HTMLVideoElement): number[] => {
  const seeks: number[] = [];

  video.addEventListener('seeking', () => seeks.push(video.currentTime));

  return seeks;
};

const isSpanStart = (seconds: number): boolean => Math.abs(seconds - 0.2) < 1e-6;

const getScrubberRoot = (): Element => {
  const scrubber = page
    .getByRole('slider', { name: 'Seek' })
    .element()
    .closest('[data-scope="slider"][data-part="root"]');

  if (!scrubber) {
    throw new Error('Expected the player scrubber.');
  }

  return scrubber;
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

/** Use a playable stand-in because the data source's empty seekable range prevents observable seeks and playback. */
const stubSpanPlayback = (
  video: HTMLVideoElement,
  { duration = 10, readyState = HTMLMediaElement.HAVE_METADATA }: { duration?: number; readyState?: number } = {}
) => {
  let currentTime = 0;
  let paused = true;
  let silentSeek = false;
  let currentReadyState = readyState;

  Object.defineProperties(video, {
    currentTime: {
      configurable: true,
      get: () => currentTime,
      set: (value: number) => {
        currentTime = value;

        if (!silentSeek) {
          video.dispatchEvent(new Event('seeking'));
        }
      },
    },
    duration: { configurable: true, get: () => duration },
    ended: { configurable: true, get: () => paused && currentTime >= duration },
    paused: { configurable: true, get: () => paused },
    readyState: { configurable: true, get: () => currentReadyState },
  });

  // `paused` flips synchronously; the `play` event is a queued task, as in a real engine.
  const play = vi.spyOn(video, 'play').mockImplementation(() => {
    paused = false;
    queueMicrotask(() => video.dispatchEvent(new Event('play')));

    return Promise.resolve();
  });
  const pause = vi.spyOn(video, 'pause').mockImplementation(() => {
    paused = true;
    video.dispatchEvent(new Event('pause'));
  });

  return {
    play,
    pauseSpy: pause,
    reload: () => {
      currentTime = 0;
      paused = true;
      silentSeek = false;
    },
    getTime: () => currentTime,
    pause: () => video.pause(),
    /** Engines fire `pause` first, then `ended`. */
    playToEnd: () => {
      currentTime = duration;
      paused = true;
      video.dispatchEvent(new Event('pause'));
      video.dispatchEvent(new Event('ended'));
    },
    scrubTo: (value: number) => {
      currentTime = value;
      video.dispatchEvent(new Event('seeking'));
    },
    /** Engines fire no `seeking` for a seek the playhead cannot act on. */
    setSilentSeek: (value: boolean) => {
      silentSeek = value;
    },
    setReadyState: (value: number) => {
      currentReadyState = value;
    },
    setTime: (value: number) => {
      currentTime = value;
    },
  };
};

const clearSpanRequest = (): void => {
  const request = getVideoSpanPlaybackRequest();

  if (request) {
    consumeVideoSpanPlaybackRequest(request.token);
  }

  const state = getVideoSpanPlaybackState();

  if (state) {
    clearVideoSpanPlaybackState(state.token);
  }
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
