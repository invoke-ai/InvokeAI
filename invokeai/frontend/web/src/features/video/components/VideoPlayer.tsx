import { Flex } from '@invoke-ai/ui-library';
import { useVideoContextMenu } from 'features/gallery/components/ContextMenu/VideoContextMenu';
import { useVideoViewerContext } from 'features/video/context/VideoViewerContext';
import { MediaController } from 'media-chrome/react';
import { memo, useRef } from 'react';
import ReactPlayer from 'react-player';
import type { VideoDTO } from 'services/api/types';

import { VideoPlayerControls } from './VideoPlayerControls';

interface VideoPlayerProps {
  videoDTO: VideoDTO;
}

export const VideoPlayer = memo(({ videoDTO }: VideoPlayerProps) => {
  const ref = useRef<HTMLDivElement>(null);
  useVideoContextMenu(videoDTO, ref);
  const { videoRef } = useVideoViewerContext();

  return (
    <Flex ref={ref} w="full" h="full" flexDirection="column" gap={4} alignItems="center" justifyContent="center">
      <MediaController
        style={{
          width: '100%',
          aspectRatio: '16/9',
        }}
      >
        <ReactPlayer
          slot="media"
          ref={videoRef}
          src={videoDTO.video_url}
          controls={false}
          width={videoDTO.width}
          height={videoDTO.height}
          pip={false}
          // crossOrigin={$authToken.get() ? 'use-credentials' : 'anonymous'}
        />

        <VideoPlayerControls />
      </MediaController>
    </Flex>
  );
});

VideoPlayer.displayName = 'VideoPlayer';
