import { canCopyImage, copyImage } from '@platform/browser/clipboard';

/** Seek far enough from zero to clear poster state and display native-resolution frame zero. */
export const FIRST_FRAME_SEEK_SECONDS = 0.0001;

export type VideoFrameCopyFailureReason =
  | 'clipboard-failed'
  | 'draw-failed'
  | 'encode-failed'
  | 'not-ready'
  | 'stale'
  | 'unsupported';

export type VideoFrameCopyResult = { ok: true } | { ok: false; reason: VideoFrameCopyFailureReason };

export const isVideoFrameReady = (video: HTMLVideoElement | null): video is HTMLVideoElement =>
  Boolean(
    video && video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && video.videoWidth > 0 && video.videoHeight > 0
  );

export const isVideoFrameCopyAvailable = (video: HTMLVideoElement | null): boolean =>
  canCopyImage() && isVideoFrameReady(video);

export const isCanvasSecurityError = (error: unknown): boolean =>
  error instanceof DOMException && error.name === 'SecurityError';

export const encodeCanvasPng = (canvas: HTMLCanvasElement): Promise<Blob | null> =>
  new Promise((resolve, reject) => {
    try {
      canvas.toBlob(resolve, 'image/png');
    } catch (error: unknown) {
      reject(error instanceof Error ? error : new Error(String(error)));
    }
  });

export const copyVideoFrame = async (
  video: HTMLVideoElement | null,
  isCurrent: () => boolean
): Promise<VideoFrameCopyResult> => {
  if (!canCopyImage()) {
    return { ok: false, reason: 'unsupported' };
  }

  if (!isVideoFrameReady(video)) {
    return { ok: false, reason: 'not-ready' };
  }

  const width = video.videoWidth;
  const height = video.videoHeight;
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext('2d');

  if (!context) {
    return { ok: false, reason: 'draw-failed' };
  }

  try {
    context.drawImage(video, 0, 0, width, height);
  } catch {
    return { ok: false, reason: 'draw-failed' };
  }

  let blob: Blob | null;

  try {
    blob = await encodeCanvasPng(canvas);
  } catch (error: unknown) {
    return { ok: false, reason: isCanvasSecurityError(error) ? 'draw-failed' : 'encode-failed' };
  }

  if (!blob) {
    return { ok: false, reason: 'encode-failed' };
  }

  if (!isCurrent()) {
    return { ok: false, reason: 'stale' };
  }

  try {
    await copyImage(blob);
  } catch {
    return { ok: false, reason: 'clipboard-failed' };
  }

  return isCurrent() ? { ok: true } : { ok: false, reason: 'stale' };
};

export const seekVideo = (video: HTMLVideoElement, seconds: number): void => {
  try {
    video.currentTime = seconds;
  } catch {
    // Some browsers throw while the seekable range is still empty; the element keeps its position.
  }
};

export interface VideoErrorRecovery {
  handleError(video: HTMLVideoElement | null, isCurrent: () => boolean): void;
  retry(video: HTMLVideoElement | null): void;
}

/**
 * Refresh the protected-media cookie once per selection (until manual retry) before surfacing a failure overlay.
 * Auth callbacks are injected so `platform` stays independent of `features/identity`.
 */
export const createVideoErrorRecovery = ({
  getAccountEpoch,
  onFailedChange,
  refreshCookie,
}: {
  getAccountEpoch: () => number;
  onFailedChange: (failed: boolean) => void;
  refreshCookie: () => Promise<boolean>;
}): VideoErrorRecovery => {
  let automaticRefreshUsed = false;
  let pendingRefresh: Promise<boolean> | null = null;

  return {
    handleError(video, isCurrent) {
      if (!video || pendingRefresh !== null) {
        return;
      }

      if (automaticRefreshUsed) {
        onFailedChange(true);
        return;
      }

      automaticRefreshUsed = true;
      const accountEpoch = getAccountEpoch();
      const refresh = refreshCookie();
      pendingRefresh = refresh;

      const finishRefresh = (refreshed: boolean): void => {
        if (pendingRefresh !== refresh) {
          return;
        }

        pendingRefresh = null;

        if (!isCurrent() || getAccountEpoch() !== accountEpoch) {
          return;
        }

        if (refreshed) {
          video.load();
        } else {
          onFailedChange(true);
        }
      };

      void refresh.then(finishRefresh, () => finishRefresh(false));
    },
    retry(video) {
      if (!video) {
        return;
      }

      automaticRefreshUsed = false;
      pendingRefresh = null;
      onFailedChange(false);
      video.load();
    },
  };
};
