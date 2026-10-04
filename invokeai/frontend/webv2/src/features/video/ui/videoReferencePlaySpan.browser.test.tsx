import type { GalleryVideoItem } from '@features/gallery';
import type { GalleryUiAdapter } from '@features/gallery/react';
import type { VideoReferenceItem, VideoSourceClip } from '@features/video/core/types';

import { ChakraProvider } from '@chakra-ui/react';
import { DndContext } from '@dnd-kit/core';
import { GalleryUiProvider } from '@features/gallery/react';
import { system } from '@theme/system';
import i18next from 'i18next';
import { act, useCallback, useEffect, useState, type ReactNode } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { VideoReferenceListField } from './VideoReferenceListField';
import { VideoSourceClipField } from './VideoSourceClipField';
import { VideoUiProvider, type VideoSpanPlaybackState, type VideoUiAdapter } from './VideoUiContext';

/** Playback converts inclusive frame bounds to seconds, including the full final frame. */

const galleryMocks = vi.hoisted(() => ({ resolve: vi.fn() }));

vi.mock('@features/gallery', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  galleryItems: { resolve: galleryMocks.resolve },
}));

const i18n = i18next.createInstance();
await i18n.use(initReactI18next).init({
  fallbackLng: 'en',
  lng: 'en',
  resources: {
    en: {
      translation: {
        widgets: {
          gallery: { picker: { dropHint: 'Drop', upload: 'Upload' } },
          video: {
            addReference: 'Add reference',
            chooseReference: 'Choose from Gallery',
            moveReferenceDown: 'Move reference down',
            moveReferenceUp: 'Move reference up',
            pauseSelection: 'Pause selection in Preview',
            playSelection: 'Play selection in Preview',
            referenceConditioningVideo: 'Video only',
            referenceConditioningVideoAudio: 'Video + audio',
            referenceDetailMatch: 'Match generation size',
            referenceDetailMax: 'Max detail',
            initialVideoBlocked: 'blocked',
            referencesHelp: 'help',
            removeReference: 'Remove reference',
            sampleLength: 'Sample Length',
            trim: 'Trim',
            trimEnd: 'End Frame',
            trimEndShort: 'End',
            trimHelp: 'help',
            trimStart: 'Start Frame',
            trimStartShort: 'Start',
            uploadImageReference: 'Upload image',
            uploadVideoReference: 'Upload video',
          },
        },
      },
    },
  },
});

let host: HTMLDivElement;
let root: Root;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const playVideoSpanInPreview = vi.fn<VideoUiAdapter['playVideoSpanInPreview']>();
const reportError = vi.fn();

/** A stand-in for the Preview player's report, driven by the test. */
let playbackState: VideoSpanPlaybackState | null = null;
const playbackListeners = new Set<() => void>();
const reportPlayback = (state: VideoSpanPlaybackState | null): void => {
  playbackState = state;

  for (const listener of playbackListeners) {
    listener();
  }
};
const pausePlayback = vi.fn(() => {
  if (playbackState) {
    reportPlayback({ ...playbackState, isPlaying: false });
  }
});

const adapter = {
  getUploadBoardId: () => 'none',
  patchValues: vi.fn(),
  playVideoSpanInPreview,
  reportError,
  touchGalleryImages: vi.fn(),
  videoSpanPlayback: {
    getState: () => playbackState,
    subscribe: (listener: () => void) => {
      playbackListeners.add(listener);

      return () => playbackListeners.delete(listener);
    },
  },
} as unknown as VideoUiAdapter;

const galleryVideoItem = {
  kind: 'video',
  name: 'clip.mp4',
} as unknown as GalleryVideoItem;

const clip = (overrides: Partial<VideoSourceClip> = {}): VideoSourceClip => ({
  endFrame: 47,
  fps: 16,
  height: 480,
  numFrames: 200,
  startFrame: 32,
  video_name: 'clip.mp4',
  width: 832,
  ...overrides,
});

const videoReference = (overrides: Partial<VideoSourceClip> = {}): VideoReferenceItem => ({
  clip: clip(overrides),
  conditioning: 'video_audio',
  kind: 'video',
});

const imageReference: VideoReferenceItem = {
  detail: 'match',
  image: { height: 512, image_name: 'still.png', width: 512 },
  kind: 'image',
};

const Harness = ({ initial }: { initial: VideoReferenceItem[] }) => {
  const [references, setReferences] = useState(initial);
  const handleChange = useCallback((update: (current: VideoReferenceItem[]) => VideoReferenceItem[]) => {
    setReferences((current) => update(current));
  }, []);

  return (
    <VideoReferenceListField
      maxImages={9}
      maxVideos={3}
      references={references}
      targetArea={null}
      onChange={handleChange}
    />
  );
};

const InitialVideoHarness = ({ disabled }: { disabled: boolean }) => {
  const [source, setSource] = useState<VideoSourceClip | null>(clip());

  return <VideoSourceClipField disabled={disabled} sourceVideo={source} onChange={setSource} />;
};

/** The reference list with its trim reachable from the test, as a slider drag would move it. */
let retrim: ((update: Partial<VideoSourceClip>) => void) | null = null;

const RetrimmableHarness = () => {
  const [references, setReferences] = useState<VideoReferenceItem[]>([videoReference()]);
  const handleChange = useCallback((update: (current: VideoReferenceItem[]) => VideoReferenceItem[]) => {
    setReferences((current) => update(current));
  }, []);

  useEffect(() => {
    retrim = (update) => {
      setReferences((current) =>
        current.map((reference) =>
          reference.kind === 'video' ? { ...reference, clip: { ...reference.clip, ...update } } : reference
        )
      );
    };
  }, []);

  return (
    <VideoReferenceListField
      maxImages={9}
      maxVideos={3}
      references={references}
      targetArea={null}
      onChange={handleChange}
    />
  );
};

const galleryAdapter = {
  gallery: { selectBoard: vi.fn(), selectItem: vi.fn(), setView: vi.fn() },
  galleryValues: {},
  notifications: { add: vi.fn(), reportError: vi.fn() },
  widgets: { openGallery: () => true, patchGalleryValues: vi.fn() },
} as unknown as GalleryUiAdapter;

const renderTree = async (element: ReactNode): Promise<void> => {
  await act(() =>
    root.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <GalleryUiProvider adapter={galleryAdapter}>
            <DndContext>
              <VideoUiProvider adapter={adapter}>{element}</VideoUiProvider>
            </DndContext>
          </GalleryUiProvider>
        </ChakraProvider>
      </I18nextProvider>
    )
  );
};

const render = (initial: VideoReferenceItem[]): Promise<void> => renderTree(<Harness initial={initial} />);

const playButtons = (): HTMLButtonElement[] => [
  ...document.querySelectorAll<HTMLButtonElement>('button[aria-label="Play selection in Preview"]'),
];

const pauseButtons = (): HTMLButtonElement[] => [
  ...document.querySelectorAll<HTMLButtonElement>('button[aria-label="Pause selection in Preview"]'),
];

const deferred = <T,>() => {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((resolvePromise) => {
    resolve = resolvePromise;
  });

  return { promise, resolve };
};

const press = async (button: HTMLButtonElement): Promise<void> => {
  await act(async () => {
    button.click();
    await Promise.resolve();
  });
};

beforeEach(() => {
  galleryMocks.resolve.mockReset().mockResolvedValue(galleryVideoItem);
  playVideoSpanInPreview.mockReset().mockReturnValue(7);
  reportError.mockReset();
  pausePlayback.mockClear();
  playbackState = null;
  retrim = null;
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});

afterEach(async () => {
  await act(() => root.unmount());
  host.remove();
});

describe('video reference span playback', () => {
  it('asks Preview for the window the trim selected, in seconds', async () => {
    await render([videoReference()]);
    const [button] = playButtons();

    expect(button).toBeDefined();
    await press(button!);

    // The lookup is cancelled with the account lifetime, like every other gallery read.
    expect(galleryMocks.resolve).toHaveBeenCalledWith(
      { kind: 'video', name: 'clip.mp4' },
      expect.any(AbortSignal) as AbortSignal
    );
    // Inclusive frames 32..47 end at the far edge of frame 47.
    expect(playVideoSpanInPreview).toHaveBeenCalledWith({
      endSeconds: 3,
      item: galleryVideoItem,
      startSeconds: 2,
    });
  });

  it('offers the control only where there is something to play', async () => {
    await render([imageReference]);

    expect(playButtons()).toHaveLength(0);
  });

  it("plays the Initial Video clip's window, and stays live while the field is disabled", async () => {
    // Editing gates must not prevent auditioning a visible clip.
    await renderTree(<InitialVideoHarness disabled />);
    const [button] = playButtons();

    expect(button).toBeDefined();
    await press(button!);

    expect(playVideoSpanInPreview).toHaveBeenCalledWith({ endSeconds: 3, item: galleryVideoItem, startSeconds: 2 });
  });

  it('refuses a second press without dropping keyboard focus', async () => {
    const pending = deferred<GalleryVideoItem>();
    galleryMocks.resolve.mockReturnValueOnce(pending.promise);
    await render([videoReference()]);
    const [button] = playButtons();

    button!.focus();
    await press(button!);

    // Native disabled would blur the focused button during the request; aria-disabled preserves focus.
    expect(button?.getAttribute('aria-disabled')).toBe('true');
    expect(button?.disabled).toBe(false);
    expect(document.activeElement).toBe(button);

    await press(button!);
    expect(galleryMocks.resolve).toHaveBeenCalledTimes(1);

    await act(async () => {
      pending.resolve(galleryVideoItem);
      await Promise.resolve();
    });

    expect(playVideoSpanInPreview).toHaveBeenCalledTimes(1);
    expect(button?.getAttribute('aria-disabled')).toBe('false');
  });

  it('turns into a pause control only for the loop it started, and stops it', async () => {
    await render([videoReference(), videoReference({ startFrame: 0, endFrame: 15 })]);
    const [first, second] = playButtons();

    // Another request's playback token is not this button's to stop.
    await act(() => reportPlayback({ isPlaying: true, pause: pausePlayback, token: 3 }));
    expect(pauseButtons()).toHaveLength(0);

    await press(first!);
    expect(playVideoSpanInPreview).toHaveBeenCalledTimes(1);

    // Preview has the loop running: this card offers to stop it, its sibling still to play.
    await act(() => reportPlayback({ isPlaying: true, pause: pausePlayback, token: 7 }));
    expect(pauseButtons()).toEqual([first]);
    expect(playButtons()).toEqual([second]);

    await press(first!);

    // A stop, not another request: the clip pauses where it is.
    expect(pausePlayback).toHaveBeenCalledTimes(1);
    expect(playVideoSpanInPreview).toHaveBeenCalledTimes(1);
    expect(galleryMocks.resolve).toHaveBeenCalledTimes(1);
    expect(pauseButtons()).toHaveLength(0);
    expect(playButtons()).toEqual([first, second]);
  });

  it('plays the trim as it now stands when pressed again after a pause', async () => {
    await renderTree(<RetrimmableHarness />);
    const [button] = playButtons();

    await press(button!);
    await act(() => reportPlayback({ isPlaying: true, pause: pausePlayback, token: 7 }));
    await press(button!);
    expect(pausePlayback).toHaveBeenCalledTimes(1);

    // A re-press starts the current trim, not the previously paused window.
    playVideoSpanInPreview.mockReturnValue(8);
    await act(() => retrim?.({ endFrame: 79, startFrame: 64 }));
    await press(button!);

    expect(playVideoSpanInPreview).toHaveBeenCalledTimes(2);
    expect(playVideoSpanInPreview).toHaveBeenLastCalledWith({ endSeconds: 5, item: galleryVideoItem, startSeconds: 4 });

    // And it is the new request's report the button now answers to.
    await act(() => reportPlayback({ isPlaying: true, pause: pausePlayback, token: 7 }));
    expect(pauseButtons()).toHaveLength(0);
    await act(() => reportPlayback({ isPlaying: true, pause: pausePlayback, token: 8 }));
    expect(pauseButtons()).toEqual([button]);
  });

  it('keeps its paused loop when a fresh press is refused', async () => {
    await render([videoReference()]);
    const [button] = playButtons();

    await press(button!);
    await act(() => reportPlayback({ isPlaying: true, pause: pausePlayback, token: 7 }));
    await press(button!);
    expect(pauseButtons()).toHaveLength(0);

    // Preview refusal leaves the first loop armed and owned by this button.
    playVideoSpanInPreview.mockReturnValueOnce(null);
    await press(button!);
    expect(playVideoSpanInPreview).toHaveBeenCalledTimes(2);

    await act(() => reportPlayback({ isPlaying: true, pause: pausePlayback, token: 7 }));
    expect(pauseButtons()).toEqual([button]);
  });

  it('follows the player when the loop is stopped or lost elsewhere', async () => {
    await render([videoReference()]);
    const [button] = playButtons();

    await press(button!);
    await act(() => reportPlayback({ isPlaying: true, pause: pausePlayback, token: 7 }));
    expect(pauseButtons()).toEqual([button]);

    // A pause from the native controls shows here too: the selection is not playing.
    await act(() => reportPlayback({ isPlaying: false, pause: pausePlayback, token: 7 }));
    expect(pauseButtons()).toHaveLength(0);

    // A native play resumes it, and this is still the loop that button started.
    await act(() => reportPlayback({ isPlaying: true, pause: pausePlayback, token: 7 }));
    expect(pauseButtons()).toEqual([button]);

    // The user scrubbed out, or Preview closed: nothing left to stop.
    await act(() => reportPlayback(null));
    expect(pauseButtons()).toHaveLength(0);
    expect(playButtons()).toEqual([button]);
  });

  it('reports a failed lookup instead of leaving the control stuck', async () => {
    galleryMocks.resolve.mockRejectedValueOnce(new Error('clip is gone'));
    await render([videoReference()]);
    const [button] = playButtons();

    await press(button!);

    expect(reportError).toHaveBeenCalledWith('clip is gone');
    expect(playVideoSpanInPreview).not.toHaveBeenCalled();
    expect(button?.disabled).toBe(false);
  });
});
