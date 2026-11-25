import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const clipboardMock = vi.hoisted(() => ({
  canCopyImage: vi.fn<() => boolean>(() => true),
  copyImage: vi.fn<(blob: Blob) => Promise<void>>(() => Promise.resolve()),
}));

vi.mock('@platform/browser/clipboard', () => ({
  canCopyImage: clipboardMock.canCopyImage,
  copyImage: clipboardMock.copyImage,
}));

import { copyVideoFrame, createVideoErrorRecovery } from './videoFrameUtils';

describe('copyVideoFrame', () => {
  const readyVideo = () =>
    ({
      readyState: 2,
      videoHeight: 480,
      videoWidth: 640,
    }) as unknown as HTMLVideoElement;

  const stubCanvas = ({
    drawImage = vi.fn(),
    getContext,
    toBlob = (cb: BlobCallback) => cb(new Blob(['png'], { type: 'image/png' })),
  }: {
    drawImage?: ReturnType<typeof vi.fn>;
    getContext?: () => unknown;
    toBlob?: (cb: BlobCallback) => void;
  } = {}) => {
    const context = { drawImage };
    const canvas = {
      getContext: getContext ?? (() => context),
      height: 0,
      toBlob,
      width: 0,
    };
    vi.stubGlobal('HTMLMediaElement', { HAVE_CURRENT_DATA: 2 });
    vi.stubGlobal('document', {
      createElement: vi.fn(() => canvas),
    });
    return { canvas, drawImage };
  };

  beforeEach(() => {
    clipboardMock.canCopyImage.mockReset().mockReturnValue(true);
    clipboardMock.copyImage.mockReset().mockResolvedValue(undefined);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('returns unsupported when clipboard image copying is unavailable', async () => {
    clipboardMock.canCopyImage.mockReturnValue(false);
    expect(await copyVideoFrame(readyVideo(), () => true)).toEqual({ ok: false, reason: 'unsupported' });
  });

  it('returns not-ready when video element is missing or lacks decoded frame dimensions', async () => {
    vi.stubGlobal('HTMLMediaElement', { HAVE_CURRENT_DATA: 2 });
    expect(await copyVideoFrame(null, () => true)).toEqual({ ok: false, reason: 'not-ready' });
    expect(
      await copyVideoFrame(
        { readyState: 1, videoHeight: 480, videoWidth: 640 } as unknown as HTMLVideoElement,
        () => true
      )
    ).toEqual({ ok: false, reason: 'not-ready' });
    expect(
      await copyVideoFrame(
        { readyState: 2, videoHeight: 0, videoWidth: 640 } as unknown as HTMLVideoElement,
        () => true
      )
    ).toEqual({ ok: false, reason: 'not-ready' });
  });

  it('returns draw-failed when 2d context is missing, drawImage throws, or canvas is tainted', async () => {
    stubCanvas({ getContext: () => null });
    expect(await copyVideoFrame(readyVideo(), () => true)).toEqual({ ok: false, reason: 'draw-failed' });

    stubCanvas({
      drawImage: vi.fn(() => {
        throw new Error('decode failed');
      }),
    });
    expect(await copyVideoFrame(readyVideo(), () => true)).toEqual({ ok: false, reason: 'draw-failed' });

    stubCanvas({
      toBlob: () => {
        throw new DOMException('tainted', 'SecurityError');
      },
    });
    expect(await copyVideoFrame(readyVideo(), () => true)).toEqual({ ok: false, reason: 'draw-failed' });
  });

  it('returns encode-failed when toBlob yields null or throws a non-security error', async () => {
    stubCanvas({ toBlob: (cb) => cb(null) });
    expect(await copyVideoFrame(readyVideo(), () => true)).toEqual({ ok: false, reason: 'encode-failed' });

    stubCanvas({
      toBlob: () => {
        throw new Error('encode crash');
      },
    });
    expect(await copyVideoFrame(readyVideo(), () => true)).toEqual({ ok: false, reason: 'encode-failed' });
  });

  it('returns stale when selection changes before or after clipboard write', async () => {
    let current = true;
    stubCanvas({
      toBlob: (cb) => {
        current = false;
        cb(new Blob(['png'], { type: 'image/png' }));
      },
    });
    expect(await copyVideoFrame(readyVideo(), () => current)).toEqual({ ok: false, reason: 'stale' });
    expect(clipboardMock.copyImage).not.toHaveBeenCalled();

    current = true;
    stubCanvas();
    clipboardMock.copyImage.mockImplementationOnce(() => {
      current = false;
      return Promise.resolve();
    });
    expect(await copyVideoFrame(readyVideo(), () => current)).toEqual({ ok: false, reason: 'stale' });
  });

  it('returns clipboard-failed on write error and ok on success', async () => {
    stubCanvas();
    clipboardMock.copyImage.mockRejectedValueOnce(new Error('denied'));
    expect(await copyVideoFrame(readyVideo(), () => true)).toEqual({ ok: false, reason: 'clipboard-failed' });

    const { canvas, drawImage } = stubCanvas();
    const video = readyVideo();
    expect(await copyVideoFrame(video, () => true)).toEqual({ ok: true });
    expect(canvas.width).toBe(640);
    expect(canvas.height).toBe(480);
    expect(drawImage).toHaveBeenCalledWith(video, 0, 0, 640, 480);
    expect(clipboardMock.copyImage).toHaveBeenCalledTimes(2);
  });
});

describe('createVideoErrorRecovery', () => {
  const makeFakeVideo = () => ({ load: vi.fn() }) as unknown as HTMLVideoElement;

  it('refreshes the protected-media cookie once per load, ignores concurrent errors, and rearms on manual retry', async () => {
    const video = makeFakeVideo();
    let failed = false;
    let resolveFirstRefresh!: (value: boolean) => void;
    const refreshCookie = vi
      .fn<() => Promise<boolean>>()
      .mockImplementationOnce(
        () =>
          new Promise<boolean>((resolve) => {
            resolveFirstRefresh = resolve;
          })
      )
      .mockResolvedValueOnce(false);

    const recovery = createVideoErrorRecovery({
      getAccountEpoch: () => 3,
      onFailedChange: (next) => {
        failed = next;
      },
      refreshCookie,
    });

    // First error starts a refresh; a second error while refresh is still in flight is ignored.
    recovery.handleError(video, () => true);
    recovery.handleError(video, () => true);
    expect(refreshCookie).toHaveBeenCalledTimes(1);
    expect(failed).toBe(false);

    resolveFirstRefresh(true);
    await vi.waitFor(() => expect(video.load).toHaveBeenCalledTimes(1));
    expect(failed).toBe(false);

    // Next error without manual retry surfaces failure instead of looping.
    recovery.handleError(video, () => true);
    expect(refreshCookie).toHaveBeenCalledTimes(1);
    expect(failed).toBe(true);

    // Manual retry clears failure, reloads, and rearms automatic refresh so the next error calls refreshCookie again.
    recovery.retry(video);
    expect(failed).toBe(false);
    expect(video.load).toHaveBeenCalledTimes(2);

    recovery.handleError(video, () => true);
    expect(refreshCookie).toHaveBeenCalledTimes(2);
    await vi.waitFor(() => expect(failed).toBe(true));
  });

  it('ignores a refresh completion when the account epoch changes mid-flight', async () => {
    const video = makeFakeVideo();
    let accountEpoch = 1;
    const onFailedChange = vi.fn();
    let resolveRefresh!: (value: boolean) => void;
    const refreshCookie = vi.fn(
      () =>
        new Promise<boolean>((resolve) => {
          resolveRefresh = resolve;
        })
    );

    const recovery = createVideoErrorRecovery({
      getAccountEpoch: () => accountEpoch,
      onFailedChange,
      refreshCookie,
    });

    recovery.handleError(video, () => true);
    accountEpoch = 2;
    resolveRefresh(true);
    await Promise.resolve();

    expect(video.load).not.toHaveBeenCalled();
    expect(onFailedChange).not.toHaveBeenCalled();
  });

  it('ignores a refresh completion when the video element is no longer current', async () => {
    const video = makeFakeVideo();
    let isCurrent = true;
    const onFailedChange = vi.fn();
    let resolveRefresh!: (value: boolean) => void;
    const refreshCookie = vi.fn(
      () =>
        new Promise<boolean>((resolve) => {
          resolveRefresh = resolve;
        })
    );

    const recovery = createVideoErrorRecovery({
      getAccountEpoch: () => 1,
      onFailedChange,
      refreshCookie,
    });

    recovery.handleError(video, () => isCurrent);
    isCurrent = false;
    resolveRefresh(true);
    await Promise.resolve();

    expect(video.load).not.toHaveBeenCalled();
    expect(onFailedChange).not.toHaveBeenCalled();
  });
});
