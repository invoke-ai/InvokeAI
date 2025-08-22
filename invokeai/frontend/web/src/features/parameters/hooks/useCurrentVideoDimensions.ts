import { useAppSelector } from 'app/store/storeHooks';
import type { AspectRatioID } from 'features/controlLayers/store/types';
import { ASPECT_RATIO_MAP, RESOLUTION_MAP } from 'features/controlLayers/store/types';
import { selectVideoAspectRatio, selectVideoResolution } from 'features/parameters/store/videoSlice';
import { useMemo } from 'react';

export const useCurrentVideoDimensions = () => {
  const videoAspectRatio = useAppSelector(selectVideoAspectRatio);
  const videoResolution = useAppSelector(selectVideoResolution);

  const currentVideoDimensions = useMemo(() => {
    // Default fallback dimensions
    const fallback = { width: 1280, height: 720 };

    if (!videoAspectRatio || !videoResolution) {
      return fallback;
    }

    // Get base resolution dimensions from the resolution tables
    let baseWidth: number;
    let baseHeight: number;

    const resolutionDims = RESOLUTION_MAP[videoResolution];
    baseWidth = resolutionDims.width;
    baseHeight = resolutionDims.height;

    // Get the aspect ratio value from the map
    const aspectRatioData = ASPECT_RATIO_MAP[videoAspectRatio as Exclude<AspectRatioID, 'Free'>];
    if (!aspectRatioData) {
      return { width: baseWidth, height: baseHeight };
    }

    const targetRatio = aspectRatioData.ratio;

    // Calculate dimensions that maintain the aspect ratio while respecting the resolution
    // We use the resolution as a constraint on the total pixel count
    const totalPixels = baseWidth * baseHeight;

    // Calculate dimensions that match the aspect ratio and approximate the target pixel count
    // width * height = totalPixels
    // width / height = targetRatio
    // Therefore: width = sqrt(totalPixels * targetRatio) and height = sqrt(totalPixels / targetRatio)
    const calculatedWidth = Math.round(Math.sqrt(totalPixels * targetRatio));
    const calculatedHeight = Math.round(Math.sqrt(totalPixels / targetRatio));

    // Ensure dimensions are even numbers (common requirement for video encoding)
    const width = calculatedWidth % 2 === 0 ? calculatedWidth : calculatedWidth + 1;
    const height = calculatedHeight % 2 === 0 ? calculatedHeight : calculatedHeight + 1;

    return { width, height };
  }, [videoAspectRatio, videoResolution]);

  return currentVideoDimensions;
};
