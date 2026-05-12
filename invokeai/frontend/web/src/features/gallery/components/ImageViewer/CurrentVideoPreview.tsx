import { Flex, Image } from '@invoke-ai/ui-library';
import { memo, useCallback, useEffect, useState } from 'react';
import type { VideoDTO } from 'services/api/types';

import { NoContentForViewer } from './NoContentForViewer';
import { VideoPlayButtonOverlay } from './VideoPlayButtonOverlay';

type Props = {
  videoDTO: VideoDTO | null;
};

/**
 * Counterpart to CurrentImagePreview for videos. Has two states:
 *
 *  - **idle**: shows the WebP first-frame thumbnail + a centered play button overlay
 *  - **playing**: swaps in an HTML5 `<video controls autoplay>` element wired to /full
 *
 * Switching the selection (so `videoDTO.video_name` changes) resets back to idle, which
 * unmounts the previous `<video>` element and stops playback cleanly.
 */
export const CurrentVideoPreview = memo(({ videoDTO }: Props) => {
  const videoName = videoDTO?.video_name ?? null;
  const [isPlaying, setIsPlaying] = useState(false);

  // Whenever the selected video changes, drop back to the still + play overlay.
  useEffect(() => {
    setIsPlaying(false);
  }, [videoName]);

  const handlePlay = useCallback(() => {
    setIsPlaying(true);
  }, []);

  if (!videoDTO) {
    return <NoContentForViewer />;
  }

  if (isPlaying) {
    return (
      <Flex width="full" height="full" alignItems="center" justifyContent="center">
        <video
          // Resolves to the /api/v1/videos/i/{name}/full route, which supports HTTP Range
          // so the browser can scrub freely.
          src={videoDTO.video_url}
          controls
          autoPlay
          // Native controls handle play/pause/scrub/volume/fullscreen.
          style={{
            maxWidth: '100%',
            maxHeight: '100%',
            borderRadius: 4,
            outline: 'none',
            background: 'black',
          }}
        />
      </Flex>
    );
  }

  // Idle: thumbnail + centered play button.
  return (
    <Flex width="full" height="full" alignItems="center" justifyContent="center" position="relative">
      <Image
        src={videoDTO.thumbnail_url}
        objectFit="contain"
        maxW="full"
        maxH="full"
        borderRadius="base"
        pointerEvents="none"
      />
      <VideoPlayButtonOverlay onClick={handlePlay} />
    </Flex>
  );
});

CurrentVideoPreview.displayName = 'CurrentVideoPreview';
