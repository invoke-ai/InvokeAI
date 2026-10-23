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

  it('wraps playback that reaches the loop window end back to its start', async () => {
    const store = createMediaElementStore();
    store.attach(video);
    await loadFixture();
    const seeks = recordSeeks();
    const onEnded = vi.fn();
    video.addEventListener('ended', onEnded);
    store.setLoopWindow({ endSeconds: 0.5, startSeconds: 0.2 });

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
    store.setLoopWindow({ endSeconds: 1, startSeconds: 0.6 });
    store.seek(0.7);

    store.play();

    await expect.poll(() => seeks.filter((time) => Math.abs(time - 0.6) < 1e-6).length, { timeout: 3000 }).toBe(1);
    await expect.poll(() => video.paused).toBe(false);
  });

  it('stops wrapping once the window is cleared, and ignores clearing a replaced window', async () => {
    const store = createMediaElementStore();
    store.attach(video);
    await loadFixture();
    const clearFirst = store.setLoopWindow({ endSeconds: 0.5, startSeconds: 0.2 });
    const clearSecond = store.setLoopWindow({ endSeconds: 0.4, startSeconds: 0.1 });
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
    store.setLoopWindow({ endSeconds: 0.3, startSeconds: 0.1 });
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
