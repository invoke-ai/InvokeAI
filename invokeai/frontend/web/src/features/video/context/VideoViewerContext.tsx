import type { PropsWithChildren, RefObject } from 'react';
import { createContext, memo, useContext, useMemo, useRef } from 'react';
import { assert } from 'tsafe';

type VideoViewerContextValue = {
  videoRef: RefObject<HTMLVideoElement>;
};

const VideoViewerContext = createContext<VideoViewerContextValue | null>(null);

export const VideoViewerContextProvider = memo((props: PropsWithChildren) => {
  const videoRef = useRef<HTMLVideoElement>(null);

  const value = useMemo(() => ({ videoRef }), [videoRef]);

  return <VideoViewerContext.Provider value={value}>{props.children}</VideoViewerContext.Provider>;
});
VideoViewerContextProvider.displayName = 'VideoViewerContextProvider';

export const useVideoViewerContext = () => {
  const value = useContext(VideoViewerContext);
  assert(value !== null, 'useVideoViewerContext must be used within a VideoViewerContextProvider');
  return value;
};
