import { Flex, Icon, IconButton } from '@invoke-ai/ui-library';
import { logger } from 'app/logging/logger';
import { $authToken } from 'app/store/nanostores/authToken';
import { useVideoContextMenu } from 'features/gallery/components/ContextMenu/VideoContextMenu';
import { toast } from 'features/toast/toast';
import { useCaptureVideoFrame } from 'features/video/hooks/useCaptureVideoFrame';
import {
  MediaControlBar,
  MediaController,
  MediaFullscreenButton,
  MediaPlayButton,
  MediaTimeDisplay,
  MediaTimeRange,
} from 'media-chrome/react';
import type { CSSProperties } from 'react';
import { memo, useCallback, useRef, useState } from 'react';
import { PiArrowsOutBold, PiCameraBold, PiPauseFill, PiPlayFill, PiSpinnerBold } from 'react-icons/pi';
import ReactPlayer from 'react-player';
import { serializeError } from 'serialize-error';
import { uploadImage } from 'services/api/endpoints/images';
import type { VideoDTO } from 'services/api/types';

const NoHoverBackground = {
  '--media-control-hover-background': 'transparent',
  '--media-text-color': 'base.200',
  '--media-font-size': '12px',
} as CSSProperties;

const log = logger('video');

export const VideoPlayer = memo(({ videoDTO }: { videoDTO: VideoDTO }) => {
  const ref = useRef<HTMLDivElement>(null);
  const reactPlayerRef = useRef<HTMLVideoElement>(null);

  const [capturing, setCapturing] = useState(false);

  const { captureFrame } = useCaptureVideoFrame();

  const onClick = useCallback(async () => {
    setCapturing(true);
    let file: File;
    try {
      const reactPlayer = reactPlayerRef.current;
      if (!reactPlayer) {
        return;
      }

      file = captureFrame(reactPlayer);
      await uploadImage({ file, image_category: 'user', is_intermediate: false, silent: true });
      toast({
        status: 'success',
        title: 'Frame saved to assets tab',
      });
    } catch (error) {
      log.error({ error: serializeError(error as Error) }, 'Failed to capture frame');
      toast({
        status: 'error',
        title: 'Failed to capture frame',
      });
    } finally {
      setCapturing(false);
    }
  }, [captureFrame]);

  useVideoContextMenu(videoDTO, ref);

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
          ref={reactPlayerRef}
          src={videoDTO.video_url}
          controls={false}
          width={videoDTO.width}
          height={videoDTO.height}
          pip={false}
          crossOrigin={$authToken.get() ? 'use-credentials' : 'anonymous'}
        />
        <MediaControlBar>
          <MediaPlayButton noTooltip={true} style={NoHoverBackground}>
            <span slot="play">
              <Icon as={PiPlayFill} boxSize={6} color="base.200" />
            </span>
            <span slot="pause">
              <Icon as={PiPauseFill} boxSize={6} color="base.200" />
            </span>
          </MediaPlayButton>
          <MediaTimeRange style={NoHoverBackground} />
          <MediaTimeDisplay showDuration style={NoHoverBackground} noToggle={true} />
          <MediaFullscreenButton noTooltip={true} style={NoHoverBackground}>
            <span slot="icon">
              <Icon as={PiArrowsOutBold} boxSize={6} color="base.200" />
            </span>
          </MediaFullscreenButton>

          <IconButton
            tooltip={capturing ? 'Capturing...' : 'Save Current Frame as Asset'}
            icon={<Icon as={capturing ? PiSpinnerBold : PiCameraBold} boxSize={6} color="base.200" />}
            size="lg"
            variant="unstyled"
            onClick={onClick}
            aria-label="Save Current Frame"
          />
        </MediaControlBar>
      </MediaController>
    </Flex>
  );
});

VideoPlayer.displayName = 'VideoPlayer';
