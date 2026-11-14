import { resolveStreamingImageSource, type StreamingImageSource } from './streamingImageSource';

export const useStreamingImageSource = ({
  fallbackImage,
  finalImage,
  liveImage,
}: {
  fallbackImage?: StreamingImageSource | null;
  finalImage?: StreamingImageSource | null;
  liveImage?: StreamingImageSource | null;
}): StreamingImageSource | null => resolveStreamingImageSource({ fallbackImage, finalImage, liveImage });
