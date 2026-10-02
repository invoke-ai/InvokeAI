import type { GalleryItemRef, GalleryVideoItem } from '@features/gallery';
import type { VideoWidgetValues } from '@features/video/core/types';
import type { ReactNode } from 'react';

import { createContext, use, useMemo } from 'react';

/** Workbench supplies this UI port because features cannot import workbench. */
export interface VideoUiAdapter {
  /** Reveal this media's board, page, and cell in Gallery and show Preview. */
  findInGallery(ref: GalleryItemRef): void;
  /** Read the selected upload board at upload time without subscribing to board changes. */
  getUploadBoardId(): string;
  patchValues(values: Partial<VideoWidgetValues>, origin?: 'user' | 'system'): void;
  /**
   * Select item in Preview and request looping this window. Returns its playback token, or null if Preview could
   * not be raised and no request was made.
   */
  playVideoSpanInPreview(span: { endSeconds: number; item: GalleryVideoItem; startSeconds: number }): number | null;
  projectId: string;
  rawValues: Record<string, unknown>;
  reportError(message: string): void;
  showPromptSyntaxHighlighting: boolean;
  touchGalleryImages(): void;
  /** What Preview is doing with the last span it was asked to play, for the button that asked. */
  videoSpanPlayback: VideoSpanPlaybackPort;
}

/**
 * Reports the armed request token, or null after disarming. isPlaying mirrors the element; pause preserves the
 * armed window for native resume.
 */
export interface VideoSpanPlaybackState {
  isPlaying: boolean;
  pause(): void;
  token: number;
}

export interface VideoSpanPlaybackPort {
  getState(): VideoSpanPlaybackState | null;
  subscribe(listener: () => void): () => void;
}

/** The adapter's callbacks, which are stable for the lifetime of a project. */
export type VideoUiActions = Pick<
  VideoUiAdapter,
  | 'findInGallery'
  | 'getUploadBoardId'
  | 'patchValues'
  | 'playVideoSpanInPreview'
  | 'reportError'
  | 'touchGalleryImages'
  | 'videoSpanPlayback'
>;

const VideoUiContext = createContext<VideoUiAdapter | null>(null);
/** Separate actions from the value-changing adapter so action-only consumers do not rerender on form edits. */
const VideoUiActionsContext = createContext<VideoUiActions | null>(null);

export const VideoUiProvider = ({ adapter, children }: { adapter: VideoUiAdapter; children: ReactNode }) => {
  const {
    findInGallery,
    getUploadBoardId,
    patchValues,
    playVideoSpanInPreview,
    reportError,
    touchGalleryImages,
    videoSpanPlayback,
  } = adapter;
  const actions = useMemo<VideoUiActions>(
    () => ({
      findInGallery,
      getUploadBoardId,
      patchValues,
      playVideoSpanInPreview,
      reportError,
      touchGalleryImages,
      videoSpanPlayback,
    }),
    [
      findInGallery,
      getUploadBoardId,
      patchValues,
      playVideoSpanInPreview,
      reportError,
      touchGalleryImages,
      videoSpanPlayback,
    ]
  );

  return (
    <VideoUiActionsContext value={actions}>
      <VideoUiContext value={adapter}>{children}</VideoUiContext>
    </VideoUiActionsContext>
  );
};

export const useVideoUi = (): VideoUiAdapter => {
  const adapter = use(VideoUiContext);

  if (!adapter) {
    throw new Error('Video UI requires an App-composed VideoUiProvider.');
  }

  return adapter;
};

export const useVideoUiActions = (): VideoUiActions => {
  const actions = use(VideoUiActionsContext);

  if (!actions) {
    throw new Error('Video UI requires an App-composed VideoUiProvider.');
  }

  return actions;
};
