import { Box, Image } from '@invoke-ai/ui-library';
import { useMediaCookieRefreshVersion } from 'features/auth/store/mediaCookieRefresh';
import { memo, useCallback, useState } from 'react';

type Props = {
  thumbnailUrl: string;
  videoUrl: string;
};

type FailedThumbnail = {
  url: string;
  mediaCookieVersion: number;
};

export const getVideoThumbnailMode = (
  thumbnailUrl: string,
  failedThumbnail: FailedThumbnail | null,
  mediaCookieVersion: number
): 'image' | 'video' =>
  thumbnailUrl === failedThumbnail?.url && mediaCookieVersion === failedThumbnail.mediaCookieVersion
    ? 'video'
    : 'image';

export const getVideoThumbnailKey = (thumbnailUrl: string, mediaCookieVersion: number) =>
  `${mediaCookieVersion}:${thumbnailUrl}`;

export const GalleryVideoThumbnail = memo(({ thumbnailUrl, videoUrl }: Props) => {
  const mediaCookieVersion = useMediaCookieRefreshVersion();
  const [failedThumbnail, setFailedThumbnail] = useState<FailedThumbnail | null>(null);
  const onError = useCallback(
    () => setFailedThumbnail({ url: thumbnailUrl, mediaCookieVersion }),
    [mediaCookieVersion, thumbnailUrl]
  );

  if (getVideoThumbnailMode(thumbnailUrl, failedThumbnail, mediaCookieVersion) === 'video') {
    return (
      <Box
        as="video"
        pointerEvents="none"
        src={videoUrl}
        preload="metadata"
        muted
        playsInline
        objectFit="contain"
        maxW="full"
        maxH="full"
        borderRadius="base"
      />
    );
  }

  return (
    <Image
      key={getVideoThumbnailKey(thumbnailUrl, mediaCookieVersion)}
      pointerEvents="none"
      src={thumbnailUrl}
      onError={onError}
      objectFit="contain"
      maxW="full"
      maxH="full"
      borderRadius="base"
    />
  );
});

GalleryVideoThumbnail.displayName = 'GalleryVideoThumbnail';
