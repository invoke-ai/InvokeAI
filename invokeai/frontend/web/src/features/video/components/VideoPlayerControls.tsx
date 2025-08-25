import { Icon, IconButton } from '@invoke-ai/ui-library';
import { logger } from 'app/logging/logger';
import { toast } from 'features/toast/toast';
import { useVideoViewerContext } from 'features/video/context/VideoViewerContext';
import { useCaptureVideoFrame } from 'features/video/hooks/useCaptureVideoFrame';
import {
  MediaControlBar,
  MediaFullscreenButton,
  MediaPlayButton,
  MediaTimeDisplay,
  MediaTimeRange,
} from 'media-chrome/react';
import type { CSSProperties } from 'react';
import { useCallback, useState } from 'react';
import { PiArrowsOutBold, PiCameraBold, PiPauseFill, PiPlayFill, PiSpinnerBold } from 'react-icons/pi';
import { serializeError } from 'serialize-error';
import { uploadImage } from 'services/api/endpoints/images';

const log = logger('video');

const NoHoverBackground = {
  '--media-control-hover-background': 'transparent',
  '--media-text-color': 'base.200',
  '--media-font-size': '12px',
} as CSSProperties;

export const VideoPlayerControls = () => {
  const { captureFrame } = useCaptureVideoFrame();
  const [capturing, setCapturing] = useState(false);
  const { $videoRef } = useVideoViewerContext();

  const onClickSaveFrame = useCallback(async () => {
    setCapturing(true);
    let file: File;
    try {
      const videoRef = $videoRef.get();
      if (!videoRef) {
        return;
      }

      file = captureFrame(videoRef);
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
  }, [captureFrame, $videoRef]);

  return (
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
        onClick={onClickSaveFrame}
        aria-label="Save Current Frame"
        isDisabled={capturing}
      />
    </MediaControlBar>
  );
};
