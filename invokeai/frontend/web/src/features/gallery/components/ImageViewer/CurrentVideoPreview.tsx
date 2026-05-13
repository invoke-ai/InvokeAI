import { Flex } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppSelector } from 'app/store/storeHooks';
import { selectShouldShowProgressInViewer } from 'features/ui/store/uiSelectors';
import { memo, useCallback, useEffect, useRef, useState } from 'react';
import type { VideoDTO } from 'services/api/types';

import { useImageViewerContext } from './context';
import { NoContentForViewer } from './NoContentForViewer';
import { ProgressImage } from './ProgressImage2';
import { ProgressIndicator } from './ProgressIndicator2';
import { VideoPlayButtonOverlay } from './VideoPlayButtonOverlay';

type Props = {
  videoDTO: VideoDTO | null;
};

/**
 * Counterpart to CurrentImagePreview for videos. A single <video> element spans both states:
 *
 *  - **idle**: muted, no controls. Without a `poster` attribute the browser decodes and
 *    displays the video's actual first frame at full resolution (much sharper than the
 *    small WebP gallery thumbnail upscaled to fit the viewer). A centered play button
 *    overlay sits on top.
 *  - **playing**: native HTML5 controls + audio. The element is the same DOM node, so the
 *    decoded buffer carries over — no reload when the user hits play.
 *
 * Changing the selected video swaps the element via `key={videoName}`, which discards the
 * old playback state cleanly.
 *
 * Mirrors CurrentImagePreview's progress overlay so denoise previews from a new render
 * appear on top of the previously-loaded video. Without this, a freshly generated render's
 * progress images had nowhere to display whenever a video was the last-selected gallery
 * item (and the user only saw the static first-frame still until the new video finished).
 */
export const CurrentVideoPreview = memo(({ videoDTO }: Props) => {
  const videoName = videoDTO?.video_name ?? null;
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const shouldShowProgressInViewer = useAppSelector(selectShouldShowProgressInViewer);
  const { $progressEvent, $progressImage } = useImageViewerContext();
  const progressEvent = useStore($progressEvent);
  const progressImage = useStore($progressImage);
  const withProgress = shouldShowProgressInViewer && progressImage !== null;

  // Whenever the selected video changes, drop back to the idle still + play overlay.
  useEffect(() => {
    setIsPlaying(false);
  }, [videoName]);

  const handlePlay = useCallback(() => {
    setIsPlaying(true);
    // The ref points at the same element we'll re-render with controls/audio; calling
    // play() here keeps the user gesture wired to playback without waiting for React.
    void videoRef.current?.play();
  }, []);

  if (!videoDTO) {
    return <NoContentForViewer />;
  }

  return (
    <Flex width="full" height="full" alignItems="center" justifyContent="center" position="relative">
      <video
        key={videoName ?? undefined}
        ref={videoRef}
        // Resolves to /api/v1/videos/i/{name}/full, which supports HTTP Range — used both
        // for first-frame decode (preload=metadata) and for scrub during playback.
        src={videoDTO.video_url}
        preload="metadata"
        muted={!isPlaying}
        playsInline
        controls={isPlaying}
        style={{
          maxWidth: '100%',
          maxHeight: '100%',
          borderRadius: 4,
          outline: 'none',
          background: 'black',
        }}
      />
      {!isPlaying && !withProgress && <VideoPlayButtonOverlay onClick={handlePlay} />}
      {withProgress && (
        <Flex w="full" h="full" position="absolute" alignItems="center" justifyContent="center" bg="base.900">
          <ProgressImage progressImage={progressImage} />
          {progressEvent && (
            <ProgressIndicator progressEvent={progressEvent} position="absolute" top={6} right={6} size={8} />
          )}
        </Flex>
      )}
    </Flex>
  );
});

CurrentVideoPreview.displayName = 'CurrentVideoPreview';
