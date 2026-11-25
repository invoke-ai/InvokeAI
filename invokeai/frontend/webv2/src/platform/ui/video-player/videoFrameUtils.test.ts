import { describe, expect, it, vi } from 'vitest';

import { createVideoErrorRecovery } from './videoFrameUtils';

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
