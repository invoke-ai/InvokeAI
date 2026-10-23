import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { createMediaElementStore } from './mediaElementStore';

// The mock backend's clip: 64×64 H.264, 10 frames at 10 fps, one second long. Vite serves the package root.
const FIXTURE_VIDEO_URL = '/scripts/mock-assets/fixture-video.mp4';

let video: HTMLVideoElement;

beforeEach(() => {
  video = document.createElement('video');
  video.muted = true;
  video.preload = 'auto';
  document.body.append(video);
});

afterEach(() => {
  video.pause();
  video.remove();
  vi.restoreAllMocks();
});

const loadFixture = async (): Promise<void> => {
  video.src = FIXTURE_VIDEO_URL;
  await expect.poll(() => video.readyState >= HTMLMediaElement.HAVE_METADATA).toBe(true);
};

/** Seek targets in the order the element reports them. */
const recordSeeks = (): number[] => {
  const seeks: number[] = [];
  video.addEventListener('seeking', () => seeks.push(video.currentTime));
  return seeks;
};

describe('createMediaElementStore', () => {
  it('reports the element through its snapshot once attached', async () => {
    const store = createMediaElementStore();
    store.attach(video);
    await loadFixture();

    await expect.poll(() => store.getSnapshot()).toMatchObject({ duration: 1, paused: true, ready: true });

    store.seek(5);
    expect(store.getSnapshot().currentTime).toBe(1);
  });

  it('wraps playback that crosses the window end back to its start', async () => {
    const store = createMediaElementStore();
    store.attach(video);
    await loadFixture();
    const seeks = recordSeeks();
    const onEnded = vi.fn();
    video.addEventListener('ended', onEnded);
    store.setPlaybackWindow({ endSeconds: 0.5, startSeconds: 0.2 }, { loop: true, restSeconds: 0 });

    store.play();

    await expect.poll(() => seeks.some((time) => Math.abs(time - 0.2) < 1e-6), { timeout: 3000 }).toBe(true);
    expect(store.getSnapshot().paused).toBe(false);
    expect(onEnded).not.toHaveBeenCalled();
  });

  it('wraps a window that ends at the media end and keeps playing', async () => {
    const store = createMediaElementStore();
    store.attach(video);
    await loadFixture();
    const seeks = recordSeeks();
    store.setPlaybackWindow({ endSeconds: 1, startSeconds: 0.6 }, { loop: true, restSeconds: 0 });
    store.seek(0.7);

    store.play();

    await expect.poll(() => seeks.filter((time) => Math.abs(time - 0.6) < 1e-6).length, { timeout: 3000 }).toBe(1);
    await expect.poll(() => video.paused).toBe(false);
  });

  it('never wraps or pauses a seek that lands past the window: playback goes on from there', async () => {
    const store = createMediaElementStore();
    store.attach(video);
    await loadFixture();
    store.setPlaybackWindow({ endSeconds: 0.5, startSeconds: 0.2 }, { loop: true, restSeconds: 0 });
    store.seek(0.25);
    store.play();
    await expect.poll(() => !video.paused && video.currentTime > 0.3).toBe(true);
    const seeks = recordSeeks();
    const onEnded = vi.fn();
    video.addEventListener('ended', onEnded);

    store.seek(0.8);

    await expect.poll(() => onEnded.mock.calls.length, { timeout: 3000 }).toBe(1);
    expect(seeks).toEqual([0.8]);
  });

  it("leaves a seek into the next window alone while the previous one is still set, as a caller's swap lags", async () => {
    const store = createMediaElementStore();
    store.attach(video);
    await loadFixture();
    store.setPlaybackWindow({ endSeconds: 0.4, startSeconds: 0.1 }, { loop: true, restSeconds: 0 });
    store.seek(0.15);
    store.play();
    await expect.poll(() => !video.paused && video.currentTime > 0.2).toBe(true);
    const seeks = recordSeeks();

    // The next window's start, past the previous end; the caller replaces the window a moment later.
    store.seek(0.6);
    await new Promise((resolve) => {
      setTimeout(resolve, 100);
    });
    store.setPlaybackWindow({ endSeconds: 0.9, startSeconds: 0.6 }, { loop: true, restSeconds: 0 });

    await expect.poll(() => seeks.filter((time) => Math.abs(time - 0.6) < 1e-6).length, { timeout: 3000 }).toBe(2);
    expect(seeks.some((time) => Math.abs(time - 0.1) < 1e-6)).toBe(false);
    expect(video.paused).toBe(false);
  });

  it('stops at the window end even when sparse samples jump from before the window to past it', () => {
    // Throttled frames and quarter-second timeupdates can step over a short window between two samples.
    let currentTime = 0;
    let paused = false;
    Object.defineProperties(video, {
      currentTime: {
        configurable: true,
        get: () => currentTime,
        set: (value: number) => {
          currentTime = value;
        },
      },
      paused: { configurable: true, get: () => paused },
    });
    vi.spyOn(video, 'pause').mockImplementation(() => {
      paused = true;
    });
    const store = createMediaElementStore();
    store.attach(video);
    store.setPlaybackWindow({ endSeconds: 0.5, startSeconds: 0.2 }, { loop: false, restSeconds: 0.45 });

    currentTime = 0.18;
    video.dispatchEvent(new Event('timeupdate'));
    currentTime = 0.52;
    video.dispatchEvent(new Event('timeupdate'));

    expect(paused).toBe(true);
    expect(currentTime).toBe(0.45);
  });

  it('rests on the last frame of a window that does not loop, and any play starts it over', async () => {
    const store = createMediaElementStore();
    store.attach(video);
    await loadFixture();
    store.setPlaybackWindow({ endSeconds: 0.5, startSeconds: 0.2 }, { loop: false, restSeconds: 0.45 });
    store.seek(0.2);
    store.play();

    await expect
      .poll(() => video.paused && !video.seeking && Math.abs(video.currentTime - 0.45) < 1e-6, {
        timeout: 3000,
      })
      .toBe(true);

    // Media keys and other callers play the element directly; the window replays all the same.
    const seeks = recordSeeks();
    await video.play();

    await expect.poll(() => seeks[0]).toBeCloseTo(0.2);
    await expect.poll(() => video.paused, { timeout: 3000 }).toBe(true);
    await expect.poll(() => Math.abs(video.currentTime - 0.45) < 1e-6).toBe(true);
  });

  it('stops wrapping once the window is cleared, and ignores clearing a replaced window', async () => {
    const store = createMediaElementStore();
    store.attach(video);
    await loadFixture();
    const clearFirst = store.setPlaybackWindow({ endSeconds: 0.5, startSeconds: 0.2 }, { loop: true, restSeconds: 0 });
    const clearSecond = store.setPlaybackWindow({ endSeconds: 0.4, startSeconds: 0.1 }, { loop: true, restSeconds: 0 });
    clearFirst();
    const seeks = recordSeeks();

    store.play();
    await expect.poll(() => seeks.some((time) => Math.abs(time - 0.1) < 1e-6), { timeout: 3000 }).toBe(true);

    clearSecond();
    const onEnded = vi.fn();
    video.addEventListener('ended', onEnded);
    await expect.poll(() => onEnded.mock.calls.length, { timeout: 3000 }).toBe(1);
  });

  it('stops listening and requesting frames once detached', async () => {
    const store = createMediaElementStore();
    const detach = store.attach(video);
    await loadFixture();
    store.setPlaybackWindow({ endSeconds: 0.3, startSeconds: 0.1 }, { loop: true, restSeconds: 0 });
    store.play();
    await expect.poll(() => store.getSnapshot().paused).toBe(false);

    detach();
    const listener = vi.fn();
    store.subscribe(listener);
    const requestFrame = vi.spyOn(window, 'requestAnimationFrame');
    const seeks = recordSeeks();
    video.currentTime = 0.4;
    const onEnded = vi.fn();
    video.addEventListener('ended', onEnded);
    await expect.poll(() => onEnded.mock.calls.length, { timeout: 3000 }).toBe(1);

    expect(store.getSnapshot()).toMatchObject({ currentTime: 0, paused: true, ready: false });
    expect(listener).not.toHaveBeenCalled();
    expect(requestFrame).not.toHaveBeenCalled();
    // Only the test's own seek: the window no longer wraps.
    expect(seeks).toEqual([0.4]);
  });
});
