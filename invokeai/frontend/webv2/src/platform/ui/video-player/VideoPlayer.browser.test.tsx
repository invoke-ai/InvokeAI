/* oxlint-disable react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import { ChakraProvider } from '@chakra-ui/react';
import { auditAccessibility } from '@platform/browser/auditAccessibility.testing';
import { system } from '@theme/system';
import { createInstance } from 'i18next';
import { act, createRef, type ReactNode, type RefObject } from 'react';
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

const AUDIT_TARGET_SIZE = { rules: { 'target-size': { enabled: true } } };

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

  it("keeps a focused control's activation keys from application hotkeys, which would cancel the press", async () => {
    // Like a workbench hotkey bound to Space: it runs on the window and prevents the key's default.
    const appHotkey = (event: KeyboardEvent) => {
      if (event.key === ' ' || event.key === 'Enter') {
        event.preventDefault();
      }
    };
    window.addEventListener('keydown', appHotkey);

    try {
      const { video } = await renderPlayer();

      await focus(page.getByRole('button', { name: 'Play' }).element());
      await userEvent.keyboard(' ');
      await expect.poll(() => video.paused).toBe(false);

      await focus(page.getByRole('button', { name: 'Loop' }).element());
      await userEvent.keyboard('{Enter}');
      await expect.element(page.getByRole('button', { name: 'Loop' })).toHaveAttribute('aria-pressed', 'false');

      await focus(page.getByRole('button', { name: 'Playback speed: 1×' }).element());
      await userEvent.keyboard('{Enter}');
      await expect.element(page.getByRole('menuitemradio', { name: '2×' })).toBeVisible();

      expect(windowKeys).toEqual([]);

      // The opened menu still picks from the keyboard (its keys are the menu's, which application hotkeys skip).
      await userEvent.keyboard('{End}{Enter}');
      await expect.poll(() => video.playbackRate).toBe(2);
    } finally {
      window.removeEventListener('keydown', appHotkey);
    }
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

  it('wraps a playback window back to its start instead of playing on', async () => {
    const seeks: number[] = [];
    const onEnded = vi.fn();
    const { video } = await renderPlayer({
      playbackWindow: { endSeconds: 0.5, startSeconds: 0.2 },
      onEnded,
      onSeeking: (event) => seeks.push(event.currentTarget.currentTime),
    });

    // The window, not the element, loops.
    expect(video.loop).toBe(false);
    await play(video);

    await expect.poll(() => seeks.some((time) => Math.abs(time - 0.2) < 1e-6), { timeout: 3000 }).toBe(true);
    expect(onEnded).not.toHaveBeenCalled();
  });

  it.each([{ from: 'the clip start', startAt: 0 }])(
    'plays a window once from $from, rests on its last frame while looping is off, and replays it',
    async ({ startAt }) => {
      const seeks: number[] = [];
      const onEnded = vi.fn();
      const onPlay = vi.fn();
      // Frames 2 to 4: the end is frame 5's start, exclusive.
      const { video } = await renderPlayer({
        defaultLoop: false,
        fps: 10,
        onEnded,
        onPlay,
        onSeeking: (event) => seeks.push(event.currentTarget.currentTime),
        playbackWindow: { endSeconds: 0.5, startSeconds: 0.2 },
      });
      const isResting = () => video.paused && !video.seeking && Math.abs(video.currentTime - 0.45) < 1e-6;
      const playButton = page.getByRole('button', { name: 'Play' });

      expect(video.loop).toBe(false);

      if (startAt > 0) {
        await act(() => {
          video.currentTime = startAt;
        });
        await expect.poll(() => video.seeking).toBe(false);
        seeks.length = 0;
      }

      await userEvent.click(playButton);

      await expect.poll(isResting, { timeout: 3000 }).toBe(true);
      await expect.element(page.getByText('Frame 4 / 9')).toBeVisible();
      expect(onPlay).toHaveBeenCalledOnce();
      expect(seeks).toHaveLength(1);
      expect(seeks[0]).toBeCloseTo(0.45);

      // The replay is a 0.3 s window, which a slow runner can finish before any poll sees it playing; the record proves
      // it happened: a second play, a rewind to the window start, and the rest again.
      await userEvent.click(playButton);

      await expect.poll(() => onPlay.mock.calls.length).toBe(2);
      await expect.poll(() => seeks.length, { timeout: 3000 }).toBe(3);
      expect(seeks[1]).toBeCloseTo(0.2);
      expect(seeks[2]).toBeCloseTo(0.45);
      await expect.poll(isResting, { timeout: 3000 }).toBe(true);
      expect(onEnded).not.toHaveBeenCalled();
    }
  );

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

  it('keeps focus in the player when a new source takes its controls away', async () => {
    const onError = vi.fn();
    await renderPlayer({ onError });
    await focus(page.getByRole('button', { name: 'Loop' }).element());

    await mount({ onError, src: MISSING_VIDEO_URL });

    await expect.poll(() => onError.mock.calls.length, { timeout: 5000 }).toBe(1);
    expect(document.activeElement).toBe(getPlayer());
  });

  // Thumbs at the ends of their tracks sit closest to the buttons beside them, where target spacing is tightest.
  it.each([
    { position: 'start', size: 'compact', volume: 1 },
    { position: 'end', size: 'full', volume: 1 },
  ] as const)(
    'passes an accessibility audit at $size size with the scrubber at its $position and volume at $volume',
    async ({ position, size, volume }) => {
      const { video } = await renderPlayer({ fps: 10, muted: false, size });

      await act(() => {
        video.volume = volume;
        video.currentTime = position === 'start' ? 0 : video.duration;
      });
      await expect
        .poll(() => page.getByRole('slider', { name: 'Seek' }).element().getAttribute('aria-valuenow'))
        .toBe(position === 'start' ? '0' : '0.9');

      if (size === 'full') {
        // The volume thumb at full volume sits beside it.
        await expect.element(page.getByRole('button', { name: 'Full screen' })).toBeVisible();
      }

      // WCAG 2.2 target spacing, which axe leaves off by default and the release journeys run.
      expect(await auditAccessibility(host, AUDIT_TARGET_SIZE)).toEqual([]);
    }
  );
});
