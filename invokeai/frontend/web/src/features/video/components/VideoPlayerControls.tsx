import { Icon, IconButton } from '@invoke-ai/ui-library';
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

const NoHoverBackground = {
//   '--media-control-hover-background': 'rgb(21, 22, 29)',
  '--media-text-color': 'base.200',
  '--media-font-size': '12px',
} as CSSProperties;

export const VideoPlayerControls = () => {
  const captureVideoFrame = useCaptureVideoFrame();
  const [capturing, setCapturing] = useState(false);
  const { videoRef } = useVideoViewerContext();

  const onClickSaveFrame = useCallback(async () => {
    setCapturing(true);
    await captureVideoFrame(videoRef.current);
    setCapturing(false);
  }, [captureVideoFrame, videoRef]);

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
        icon={<Icon as={capturing ? PiSpinnerBold : PiCameraBold} boxSize={6} h={10}color="base.200" />}
        size="lg"
        variant="unstyled"
        onClick={onClickSaveFrame}
        aria-label="Save Current Frame"
        isDisabled={capturing}
        _disabled={{
          background: 'rgba(20, 20, 30, 0.7)',
        }}
        height="100%"
        backgroundColor="rgba(20, 20, 30, 0.7)"
        pb={3}
      />
    </MediaControlBar>
  );
};
