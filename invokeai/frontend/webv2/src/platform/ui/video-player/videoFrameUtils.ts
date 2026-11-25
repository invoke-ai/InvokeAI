import { canCopyImage } from '@platform/browser/clipboard';

/** Seek far enough from zero to clear poster state and display native-resolution frame zero. */
export const FIRST_FRAME_SEEK_SECONDS = 0.0001;

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
