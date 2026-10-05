import { resolveStreamingImageSource, type StreamingImageSource } from './streamingImageSource';

export const useStreamingImageSource = ({
  fallbackImage,
  finalImage,
  heldLiveImage,
  liveImage,
}: {
  fallbackImage?: StreamingImageSource | null;
  finalImage?: StreamingImageSource | null;
  /** The last live frame kept up while no current live frame exists. */
  heldLiveImage?: StreamingImageSource | null;
  liveImage?: StreamingImageSource | null;
}): StreamingImageSource | null => resolveStreamingImageSource({ fallbackImage, finalImage, heldLiveImage, liveImage });
