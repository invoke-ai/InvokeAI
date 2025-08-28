import type { MediaStateOwner } from 'media-chrome/dist/media-store/state-mediator.js';
import { type Atom, atom } from 'nanostores';
import type { PropsWithChildren } from 'react';
import { createContext, memo, useCallback, useContext, useMemo, useState } from 'react';
import { assert } from 'tsafe';

type VideoViewerContextValue = {
  $videoRef: Atom<MediaStateOwner | null>;
  setVideoRef: (ref: MediaStateOwner | null) => void;
  onLoadVideo: () => void;
};

const VideoViewerContext = createContext<VideoViewerContextValue | null>(null);

export const VideoViewerContextProvider = memo((props: PropsWithChildren) => {
  const $videoRef = useState(() => atom<MediaStateOwner | null>(null))[0];

  const setVideoRef = useCallback(
    (ref: MediaStateOwner | null) => {
      $videoRef.set(ref);
    },
    [$videoRef]
  );

  const onLoadVideo = useCallback(() => {
    // Reset video ref when loading new video
    $videoRef.set(null);
  }, [$videoRef]);

  const value = useMemo(() => ({ $videoRef, setVideoRef, onLoadVideo }), [$videoRef, setVideoRef, onLoadVideo]);

  return <VideoViewerContext.Provider value={value}>{props.children}</VideoViewerContext.Provider>;
});
VideoViewerContextProvider.displayName = 'VideoViewerContextProvider';

export const useVideoViewerContext = () => {
  const value = useContext(VideoViewerContext);
  assert(value !== null, 'useVideoViewerContext must be used within a VideoViewerContextProvider');
  return value;
};
