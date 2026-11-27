import { Flex, Text } from '@chakra-ui/react';
import { Button } from '@platform/ui/Button';
import { useCallback, useRef, useState, type CSSProperties, type MouseEvent } from 'react';
import { useTranslation } from 'react-i18next';

import { createVideoErrorRecovery, type VideoErrorRecovery } from './videoFrameUtils';

const VIDEO_FAILURE_POSTER_STYLE: CSSProperties = {
  display: 'block',
  height: '100%',
  inset: 0,
  objectFit: 'contain',
  position: 'absolute',
  width: '100%',
};

export const useVideoErrorRecovery = ({
  getAccountEpoch,
  refreshCookie,
}: {
  getAccountEpoch: () => number;
  refreshCookie: () => Promise<boolean>;
}) => {
  const [hasFailed, setHasFailed] = useState(false);
  const recoveryRef = useRef<VideoErrorRecovery | null>(null);

  const getRecovery = useCallback(() => {
    if (!recoveryRef.current) {
      recoveryRef.current = createVideoErrorRecovery({
        getAccountEpoch,
        onFailedChange: setHasFailed,
        refreshCookie,
      });
    }
    return recoveryRef.current;
  }, [getAccountEpoch, refreshCookie]);

  const handleError = useCallback(
    (video: HTMLVideoElement | null, isCurrent: () => boolean) => {
      getRecovery().handleError(video, isCurrent);
    },
    [getRecovery]
  );

  const retry = useCallback(
    (video: HTMLVideoElement | null) => {
      getRecovery().retry(video);
    },
    [getRecovery]
  );

  return { handleError, hasFailed, retry };
};

export interface VideoFailureOverlayProps {
  height: number;
  onRetry: (event: MouseEvent<HTMLButtonElement>) => void;
  poster: string;
  size: 'compact' | 'full';
  testId?: string;
  width: number;
}

export const VideoFailureOverlay = ({ height, onRetry, poster, size, testId, width }: VideoFailureOverlayProps) => {
  const { t } = useTranslation();
  const isCompact = size === 'compact';

  return (
    <>
      <img
        aria-hidden="true"
        alt=""
        draggable={false}
        height={height}
        src={poster}
        style={VIDEO_FAILURE_POSTER_STYLE}
        width={width}
      />
      <Flex
        align="center"
        bg="blackAlpha.700"
        data-testid={testId}
        direction="column"
        gap={isCompact ? '2.5' : '3'}
        inset="0"
        justify="center"
        position="absolute"
        px={isCompact ? '3' : undefined}
        textAlign={isCompact ? 'center' : undefined}
      >
        <Text color="white" fontSize={isCompact ? 'sm' : 'lg'} fontWeight="semibold">
          {t('widgets.preview.videoFailed')}
        </Text>
        <Button aria-label={t('widgets.preview.videoRetry')} size={isCompact ? 'sm' : 'lg'} onClick={onRetry}>
          {t('widgets.preview.videoRetry')}
        </Button>
      </Flex>
    </>
  );
};
