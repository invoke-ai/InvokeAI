import { Box, type BoxProps } from '@chakra-ui/react';
import { useMemo, type CSSProperties, type ReactNode } from 'react';

import type { StreamingImageSource } from './streamingImageSource';

import { useStreamingImageSource } from './useStreamingImageSource';

export const StreamingImageFrame = ({
  badge,
  fallbackImage,
  finalImage,
  fit = 'cover',
  liveImage,
  shouldAntialiasLiveImage = true,
  children,
  ...boxProps
}: {
  badge?: ReactNode | ((source: StreamingImageSource) => ReactNode);
  fallbackImage?: StreamingImageSource | null;
  finalImage?: StreamingImageSource | null;
  fit?: 'contain' | 'cover';
  liveImage?: StreamingImageSource | null;
  shouldAntialiasLiveImage?: boolean;
} & BoxProps) => {
  const source = useStreamingImageSource({ fallbackImage, finalImage, liveImage });
  const renderedBadge: ReactNode = typeof badge === 'function' ? (source ? badge(source) : null) : badge;
  const imageStyle = useMemo<CSSProperties>(
    () => ({
      display: 'block',
      height: '100%',
      imageRendering: source?.kind === 'live' && !shouldAntialiasLiveImage ? 'pixelated' : 'auto',
      inset: 0,
      maxWidth: 'none',
      objectFit: fit,
      position: 'absolute',
      width: '100%',
    }),
    [fit, shouldAntialiasLiveImage, source?.kind]
  );

  return (
    <Box overflow="hidden" position="relative" {...boxProps}>
      {source ? <img alt={source.alt} draggable={false} src={source.src} style={imageStyle} /> : children}
      {renderedBadge}
    </Box>
  );
};
