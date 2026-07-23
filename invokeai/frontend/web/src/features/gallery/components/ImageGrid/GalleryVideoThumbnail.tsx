import { Box, Image } from '@invoke-ai/ui-library';
import { getMediaUrl, useMediaCookieRefreshVersion } from 'features/auth/store/mediaCookieRefresh';
import type { SyntheticEvent } from 'react';
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
  const refreshedThumbnailUrl = getMediaUrl(thumbnailUrl, mediaCookieVersion);
  const refreshedVideoUrl = getMediaUrl(videoUrl, mediaCookieVersion);
  const [failedThumbnail, setFailedThumbnail] = useState<FailedThumbnail | null>(null);
  const onError = useCallback(
    () => setFailedThumbnail({ url: thumbnailUrl, mediaCookieVersion }),
    [mediaCookieVersion, thumbnailUrl]
  );

  // With preload="metadata" some browsers populate dimensions/duration but don't decode
  // and paint the first frame until playback or a seek — the tile would just show its
  // black background. Mirror CurrentVideoPreview's near-zero seek to nudge the decoder.
  const onLoadedMetadata = useCallback((e: SyntheticEvent<HTMLElement>) => {
    const el = e.currentTarget as HTMLVideoElement;
    if (el.currentTime === 0) {
      try {
        el.currentTime = 0.0001;
      } catch {
        // Some browsers throw if metadata isn't fully ready yet; harmless.
      }
    }
  }, []);

  if (getVideoThumbnailMode(thumbnailUrl, failedThumbnail, mediaCookieVersion) === 'video') {
    return (
      <Box
        as="video"
        pointerEvents="none"
        src={refreshedVideoUrl}
        preload="metadata"
        muted
        playsInline
        objectFit="contain"
        maxW="full"
        maxH="full"
        borderRadius="base"
        onLoadedMetadata={onLoadedMetadata}
      />
    );
  }

  return (
    <Image
      key={getVideoThumbnailKey(thumbnailUrl, mediaCookieVersion)}
      pointerEvents="none"
      src={refreshedThumbnailUrl}
      onError={onError}
      objectFit="contain"
      maxW="full"
      maxH="full"
      borderRadius="base"
    />
  );
});

GalleryVideoThumbnail.displayName = 'GalleryVideoThumbnail';
