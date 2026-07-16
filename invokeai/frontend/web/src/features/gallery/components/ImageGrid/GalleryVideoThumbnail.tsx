import { Box, Image } from '@invoke-ai/ui-library';
import { memo, useCallback, useState } from 'react';

type Props = {
  thumbnailUrl: string;
  videoUrl: string;
};

export const getVideoThumbnailMode = (thumbnailUrl: string, failedThumbnailUrl: string | null): 'image' | 'video' =>
  thumbnailUrl === failedThumbnailUrl ? 'video' : 'image';

export const GalleryVideoThumbnail = memo(({ thumbnailUrl, videoUrl }: Props) => {
  const [failedThumbnailUrl, setFailedThumbnailUrl] = useState<string | null>(null);
  const onError = useCallback(() => setFailedThumbnailUrl(thumbnailUrl), [thumbnailUrl]);

  if (getVideoThumbnailMode(thumbnailUrl, failedThumbnailUrl) === 'video') {
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
