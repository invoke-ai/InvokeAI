/* oxlint-disable react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import { ChakraProvider } from '@chakra-ui/react';
import { auditAccessibility } from '@platform/browser/auditAccessibility.testing';
import { system } from '@theme/system';
import { createInstance } from 'i18next';
import { act, Activity, createRef, type ReactNode, type RefObject } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

import { VideoPlayer, type VideoPlayerProps } from './VideoPlayer';

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

// The mock backend's clip: 64×64 H.264, 10 frames at 10 fps, one second long. Vite serves the package root.
const FIXTURE_VIDEO_URL = '/scripts/mock-assets/fixture-video.mp4';
const MISSING_VIDEO_URL = '/missing-video.mp4';

const i18n = createInstance();
await i18n.use(initReactI18next).init({
  fallbackLng: 'en',
  interpolation: { escapeValue: false },
  lng: 'en',
  resources: { en: { translation: await fetch('/locales/en.json').then((response) => response.json()) } },
});

let host: HTMLDivElement;
let root: Root;
let windowKeys: string[];
const recordWindowKey = (event: KeyboardEvent) => windowKeys.push(event.key);

beforeEach(() => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  windowKeys = [];
  window.addEventListener('keydown', recordWindowKey);
});

afterEach(async () => {
  window.removeEventListener('keydown', recordWindowKey);
  await act(() => root.unmount());
  host.remove();
});

type TestProps = Partial<VideoPlayerProps> & {
  videoRef?: RefObject<HTMLVideoElement | null>;
  wrap?: (player: ReactNode) => ReactNode;
};

/** Renders (or re-renders) the player beside an outside button, without waiting for media. */
const mount = async ({ videoRef, wrap = (player) => player, ...props }: TestProps = {}) => {
  await act(() => {
    root.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <button type="button">Outside</button>
          <div style={{ height: props.size === 'compact' ? 128 : 270, width: 480 }}>
            {wrap(<VideoPlayer ref={videoRef} aria-label="Clip" muted src={FIXTURE_VIDEO_URL} {...props} />)}
          </div>
        </ChakraProvider>
      </I18nextProvider>
    );
  });
};

const renderPlayer = async ({ videoRef = createRef<HTMLVideoElement>(), ...props }: TestProps = {}) => {
  await mount({ ...props, videoRef });
  const video = videoRef.current;

  if (!video) {
    throw new Error('The player did not render its video.');
  }

  // Controls stay disabled until metadata loads.
  await expect.element(page.getByRole('button', { name: 'Loop' })).toBeEnabled();

  return { video };
};

const getPlayer = () => page.getByRole('group', { name: 'Clip' }).element() as HTMLElement;

const focus = async (element: Element) => {
  await act(() => (element as HTMLElement).focus());
};

const play = async (video: HTMLVideoElement) => {
  await userEvent.click(page.getByRole('button', { name: 'Play' }));
  await expect.poll(() => video.paused).toBe(false);
};

/** Holds still on the clip's playback for a moment, long enough for a wrongly paused element to show it. */
const settle = () =>
  new Promise<void>((resolve) => {
    setTimeout(resolve, 200);
  });

/** Dispatches a mouse pointer event at a fraction of the scrubber's width; the slider tracks moves on the document. */
const scrubPointer = async (
  type: 'pointerdown' | 'pointermove' | 'pointerup',
  fraction: number,
  modifiers: Pick<PointerEventInit, 'ctrlKey'> = {}
) => {
  const control = getPlayer().querySelector('[data-part="control"]');

  if (!control) {
    throw new Error('The scrubber has no control.');
  }

  const rect = control.getBoundingClientRect();
  await act(() => {
    (type === 'pointerdown' ? control : document).dispatchEvent(
      new PointerEvent(type, {
        bubbles: true,
        button: 0,
        buttons: type === 'pointerup' ? 0 : 1,
        cancelable: true,
        clientX: rect.left + rect.width * fraction,
        clientY: rect.top + rect.height / 2,
        isPrimary: true,
        pointerId: 1,
        pointerType: 'mouse',
        ...modifiers,
      })
    );
  });
};

describe('VideoPlayer', () => {
  it('plays and pauses from its button, Space on the player, and K', async () => {
    const { video } = await renderPlayer();

    await play(video);
    await expect.element(page.getByRole('button', { name: 'Pause' })).toBeVisible();

    await focus(getPlayer());
    await userEvent.keyboard(' ');
    await expect.poll(() => video.paused).toBe(true);

    await userEvent.keyboard('k');
    await expect.poll(() => video.paused).toBe(false);
  });

  it('leaves Space to a focused button, so it toggles once', async () => {
    const { video } = await renderPlayer();

    await focus(page.getByRole('button', { name: 'Play' }).element());
    await userEvent.keyboard(' ');

    await expect.poll(() => video.paused).toBe(false);
  });

  it('toggles once for a double-click on the video', async () => {
    const { video } = await renderPlayer();

    await userEvent.dblClick(video);
    await settle();

    expect(video.paused).toBe(false);
  });

  it('loops by default and toggles looping from L and its button', async () => {
    const { video } = await renderPlayer();
    const loopButton = page.getByRole('button', { name: 'Loop' });

    expect(video.loop).toBe(true);
    await expect.element(loopButton).toHaveAttribute('aria-pressed', 'true');

    await focus(getPlayer());
    await userEvent.keyboard('l');
    await expect.element(loopButton).toHaveAttribute('aria-pressed', 'false');
    expect(video.loop).toBe(false);

    await userEvent.click(loopButton);
    await expect.element(loopButton).toHaveAttribute('aria-pressed', 'true');
    expect(video.loop).toBe(true);
  });

  it('reports loop changes without changing a controlled loop', async () => {
    const onLoopChange = vi.fn();
    const { video } = await renderPlayer({ loop: false, onLoopChange });
    const loopButton = page.getByRole('button', { name: 'Loop' });

    await userEvent.click(loopButton);

    expect(onLoopChange).toHaveBeenCalledExactlyOnceWith(true);
    await expect.element(loopButton).toHaveAttribute('aria-pressed', 'false');
    expect(video.loop).toBe(false);
  });

  it('wraps a loop range back to its start instead of playing on', async () => {
    const seeks: number[] = [];
    const onEnded = vi.fn();
    const { video } = await renderPlayer({
      loopRange: { endSeconds: 0.5, startSeconds: 0.2 },
      onEnded,
      onSeeking: (event) => seeks.push(event.currentTarget.currentTime),
    });

    // The window, not the element, loops.
    expect(video.loop).toBe(false);
    await play(video);

    await expect.poll(() => seeks.some((time) => Math.abs(time - 0.2) < 1e-6), { timeout: 3000 }).toBe(true);
    expect(onEnded).not.toHaveBeenCalled();
  });

  it('plays a loop range through to the end while looping is off', async () => {
    const seeks: number[] = [];
    const onEnded = vi.fn();
    const { video } = await renderPlayer({
      defaultLoop: false,
      loopRange: { endSeconds: 0.5, startSeconds: 0.2 },
      onEnded,
      onSeeking: (event) => seeks.push(event.currentTarget.currentTime),
    });

    await play(video);

    await expect.poll(() => onEnded.mock.calls.length, { timeout: 3000 }).toBe(1);
    expect(seeks).toEqual([]);
  });

  it('loops the whole clip for a collapsed loop range', async () => {
    const { video } = await renderPlayer({ loopRange: { endSeconds: 0.4, startSeconds: 0.4 } });

    expect(video.loop).toBe(true);
  });

  it('steps one frame at a time with a frame rate and shows the frame', async () => {
    const { video } = await renderPlayer({ fps: 10 });

    await expect.element(page.getByText('Frame 0 / 9')).toBeVisible();
    await focus(getPlayer());
    await userEvent.keyboard('.');
    await expect.element(page.getByText('Frame 1 / 9')).toBeVisible();
    expect(video.currentTime).toBeCloseTo(0.15);

    await userEvent.click(page.getByRole('button', { name: 'Next frame' }));
    await expect.element(page.getByText('Frame 2 / 9')).toBeVisible();

    await focus(getPlayer());
    await userEvent.keyboard(',,,');
    await expect.element(page.getByText('Frame 0 / 9')).toBeVisible();
    expect(video.currentTime).toBeCloseTo(0.05);
  });

  it('never loses a frame step pressed while earlier seeks are still pending', async () => {
    const { video } = await renderPlayer({ fps: 10 });

    // Faster than any key repeat: every press lands while all the earlier seeks are pending.
    await act(() => {
      for (let press = 0; press < 12; press += 1) {
        getPlayer().dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, cancelable: true, key: '.' }));
      }
    });

    await expect.element(page.getByText('Frame 9 / 9')).toBeVisible();
    expect(video.currentTime).toBeCloseTo(0.95);
  });

  it('puts the scrubber thumb at the end of the track on the last frame', async () => {
    await renderPlayer({ fps: 10 });
    const seek = page.getByRole('slider', { name: 'Seek' });

    await focus(getPlayer());

    for (let frame = 1; frame <= 9; frame += 1) {
      await userEvent.keyboard('.');
      await expect.element(page.getByText(`Frame ${frame} / 9`)).toBeVisible();
    }

    await expect.element(seek).toHaveAttribute('aria-valuenow', seek.element().getAttribute('aria-valuemax'));
    await expect.element(seek).toHaveAttribute('aria-valuetext', 'Frame 9, last 9');
  });

  it('takes a given frame count over the one the duration implies', async () => {
    const { video } = await renderPlayer({ fps: 10, frameCount: 5 });

    await expect.element(page.getByText('Frame 0 / 4')).toBeVisible();
    await focus(getPlayer());
    await userEvent.keyboard('.'.repeat(6));

    await expect.element(page.getByText('Frame 4 / 4')).toBeVisible();
    expect(video.currentTime).toBeCloseTo(0.45);
  });

  it('pauses while the scrubber is dragged, lands where it points, and resumes after', async () => {
    const { video } = await renderPlayer();
    await play(video);

    await scrubPointer('pointerdown', 0.3);
    await scrubPointer('pointermove', 0.537);
    await settle();

    expect(video.paused).toBe(true);
    // Finer than the 0.1 s arrow-key step.
    expect(video.currentTime).toBeCloseTo(0.537, 2);
    expect(getPlayer().hasAttribute('data-controls-hidden')).toBe(false);

    await scrubPointer('pointerup', 0.537);
    await expect.poll(() => video.paused).toBe(false);
  });

  it('leaves playback alone for a modified click on the scrubber, which starts no drag', async () => {
    const { video } = await renderPlayer();
    await play(video);

    await scrubPointer('pointerdown', 0.5, { ctrlKey: true });
    await scrubPointer('pointerup', 0.5, { ctrlKey: true });
    await settle();

    expect(video.paused).toBe(false);
    await userEvent.unhover(getPlayer());
    await expect.poll(() => getPlayer().hasAttribute('data-controls-hidden')).toBe(true);
  });

  it('drops a drag the player cancels by changing source, without pinning the controls', async () => {
    const { video } = await renderPlayer();
    await play(video);
    await scrubPointer('pointerdown', 0.3);
    expect(video.paused).toBe(true);

    await mount({ src: `${FIXTURE_VIDEO_URL}?second` });
    await expect.element(page.getByRole('button', { name: 'Loop' })).toBeEnabled();
    // The slider cancelled the drag, so this release ends nothing and resumes nothing.
    await scrubPointer('pointerup', 0.3);
    await settle();
    expect(video.paused).toBe(true);

    await play(video);
    await userEvent.unhover(getPlayer());
    await expect.poll(() => getPlayer().hasAttribute('data-controls-hidden')).toBe(true);

    // A new drag starts from the current state: playing, so it resumes.
    await scrubPointer('pointerdown', 0.3);
    expect(video.paused).toBe(true);
    await scrubPointer('pointerup', 0.3);
    await expect.poll(() => video.paused).toBe(false);
  });

  it('moves the scrubber by a coarser step from arrow keys than a drag', async () => {
    const { video } = await renderPlayer();
    const seek = page.getByRole('slider', { name: 'Seek' });

    await focus(seek.element());
    await userEvent.keyboard('{ArrowRight}{ArrowRight}');
    await expect.poll(() => video.currentTime).toBeCloseTo(0.2);

    await userEvent.keyboard('{ArrowLeft}');
    await expect.poll(() => video.currentTime).toBeCloseTo(0.1);
  });

  it('hides idle controls while playing until keyboard focus or a pause brings them back', async () => {
    const { video } = await renderPlayer();
    await play(video);
    const controls = getPlayer().querySelector('[data-video-player-controls]');

    if (!controls) {
      throw new Error('The player has no controls.');
    }

    await userEvent.unhover(getPlayer());
    await expect.poll(() => getPlayer().hasAttribute('data-controls-hidden')).toBe(true);
    await expect.poll(() => getComputedStyle(controls).opacity).toBe('0');

    await userEvent.tab();
    await expect.poll(() => getComputedStyle(controls).opacity).toBe('1');

    await userEvent.keyboard('k');
    await expect.poll(() => video.paused).toBe(true);
    expect(getPlayer().hasAttribute('data-controls-hidden')).toBe(false);
  });

  it('offers no frame stepping without a frame rate', async () => {
    const { video } = await renderPlayer();

    expect(page.getByRole('button', { name: 'Next frame' }).query()).toBeNull();
    await focus(getPlayer());
    await userEvent.keyboard('.');
    expect(video.currentTime).toBe(0);
  });

  it('changes speed and mute from the bar', async () => {
    const { video } = await renderPlayer({ muted: false });

    await userEvent.click(page.getByRole('button', { name: 'Playback speed: 1×' }));
    await userEvent.click(page.getByRole('menuitemradio', { name: '2×' }));
    await expect.poll(() => video.playbackRate).toBe(2);
    await expect.element(page.getByRole('button', { name: 'Playback speed: 2×' })).toBeVisible();

    await focus(getPlayer());
    await userEvent.keyboard('m');
    await expect.poll(() => video.muted).toBe(true);
    await expect.element(page.getByRole('button', { name: 'Unmute' })).toBeVisible();
  });

  it('ignores its shortcuts while focus is outside the player', async () => {
    const { video } = await renderPlayer({ fps: 10 });

    await focus(page.getByRole('button', { name: 'Outside' }).element());
    await userEvent.keyboard('kl.');

    expect(video.paused).toBe(true);
    expect(video.loop).toBe(true);
    expect(video.currentTime).toBe(0);
    expect(windowKeys).toEqual(['k', 'l', '.']);
  });

  it('keeps its shortcuts from the page even where they do nothing', async () => {
    const { video } = await renderPlayer({ muted: false, size: 'compact' });

    await focus(getPlayer());
    // No frame rate, and compact has no volume or fullscreen.
    await userEvent.keyboard(',.mf');

    expect(video.muted).toBe(false);
    expect(windowKeys).toEqual([]);
  });

  it('leaves modified keys to the page', async () => {
    const { video } = await renderPlayer();

    await focus(getPlayer());
    await userEvent.keyboard('{Shift>}k{/Shift}');

    expect(video.paused).toBe(true);
    expect(windowKeys.map((key) => key.toLowerCase())).toEqual(['shift', 'k']);
  });

  it('leaves a media error to the caller, offers no controls, and keeps its shortcuts', async () => {
    const onError = vi.fn();
    await mount({ onError, src: MISSING_VIDEO_URL });

    await expect.poll(() => onError.mock.calls.length, { timeout: 5000 }).toBe(1);
    expect(page.getByRole('button', { name: 'Play' }).query()).toBeNull();
    expect(page.getByRole('slider', { name: 'Seek' }).query()).toBeNull();

    await focus(getPlayer());
    await userEvent.keyboard(' fmk');
    expect(windowKeys).toEqual([]);
  });

  it('keeps focus in the player when a new source takes its controls away', async () => {
    const onError = vi.fn();
    await renderPlayer({ onError });
    await focus(page.getByRole('button', { name: 'Loop' }).element());

    await mount({ onError, src: MISSING_VIDEO_URL });

    await expect.poll(() => onError.mock.calls.length, { timeout: 5000 }).toBe(1);
    expect(document.activeElement).toBe(getPlayer());
  });

  it('pauses when hidden', async () => {
    const videoRef = createRef<HTMLVideoElement>();
    const { video } = await renderPlayer({ videoRef, wrap: (player) => <Activity mode="visible">{player}</Activity> });
    await play(video);

    await mount({ videoRef, wrap: (player) => <Activity mode="hidden">{player}</Activity> });

    expect(video.paused).toBe(true);
  });

  it('keeps playing when a caller passes a new ref callback each render', async () => {
    const elements: (HTMLVideoElement | null)[] = [];
    await mount({ ref: (element) => void elements.push(element) });
    await expect.element(page.getByRole('button', { name: 'Loop' })).toBeEnabled();
    const video = elements.at(-1);

    if (!video) {
      throw new Error('The player did not render its video.');
    }

    await play(video);
    await mount({ ref: (element) => void elements.push(element) });
    await mount({ ref: (element) => void elements.push(element) });
    await settle();

    expect(video.paused).toBe(false);
    expect(elements.at(-1)).toBe(video);
  });

  it('stops tracking the element once unmounted', async () => {
    const videoRef = createRef<HTMLVideoElement>();
    const { video } = await renderPlayer({ loopRange: { endSeconds: 0.3, startSeconds: 0.1 }, videoRef });
    await play(video);

    await act(() => root.unmount());
    root = createRoot(host);
    // Unmounting paused it; play the retained element to see whether anything still drives it.
    expect(video.paused).toBe(true);
    const requestFrame = vi.spyOn(window, 'requestAnimationFrame');
    const seeks: number[] = [];
    video.addEventListener('seeking', () => seeks.push(video.currentTime));
    video.currentTime = 0.4;
    void video.play();
    const onEnded = vi.fn();
    video.addEventListener('ended', onEnded);

    // A live loop window would have wrapped to 0.1 instead of reaching the end.
    await expect.poll(() => onEnded.mock.calls.length, { timeout: 3000 }).toBe(1);
    expect(videoRef.current).toBeNull();
    expect(seeks).toEqual([0.4]);
    expect(requestFrame).not.toHaveBeenCalled();
    requestFrame.mockRestore();
    video.pause();
  });

  it.each(['full', 'compact'] as const)('passes an accessibility audit at %s size', async (size) => {
    await renderPlayer({ fps: 10, size });

    expect(await auditAccessibility(host)).toEqual([]);
  });
});
