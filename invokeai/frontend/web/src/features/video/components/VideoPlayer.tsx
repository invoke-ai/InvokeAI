import { Flex } from '@invoke-ai/ui-library';
import { $authToken } from 'app/store/nanostores/authToken';
import { useVideoContextMenu } from 'features/gallery/components/ContextMenu/VideoContextMenu';
import { useVideoViewerContext } from 'features/video/context/VideoViewerContext';
import type { MediaStateOwner } from 'media-chrome/dist/media-store/state-mediator.js';
import { MediaController } from 'media-chrome/react';
import { memo, useCallback, useRef } from 'react';
import ReactPlayer from 'react-player';
import type { VideoDTO } from 'services/api/types';

import { VideoPlayerControls } from './VideoPlayerControls';

interface VideoPlayerProps {
  videoDTO: VideoDTO;
}

export const VideoPlayer = memo(({ videoDTO }: VideoPlayerProps) => {
  const ref = useRef<HTMLDivElement>(null);
  const reactPlayerRef = useRef<MediaStateOwner | null>(null);
  useVideoContextMenu(videoDTO, ref);
  const { setVideoRef } = useVideoViewerContext();

  const handleMediaRef = useCallback(
    (mediaEl: MediaStateOwner | null | undefined) => {
      reactPlayerRef.current = mediaEl || null;
      setVideoRef(mediaEl || null);
    },
    [setVideoRef]
  );

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
          ref={handleMediaRef}
          src={videoDTO.video_url}
          controls={false}
          width={videoDTO.width}
          height={videoDTO.height}
          pip={false}
          crossOrigin={$authToken.get() ? 'use-credentials' : 'anonymous'}
        />

        <VideoPlayerControls />
      </MediaController>
    </Flex>
  );
});

VideoPlayer.displayName = 'VideoPlayer';
