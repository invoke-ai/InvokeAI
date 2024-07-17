import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import type { Dimensions } from 'features/controlLayers/store/types';
import { selectGallerySlice } from 'features/gallery/store/gallerySlice';
import type { ComparisonFit } from 'features/gallery/store/types';
import type { ImageDTO } from 'services/api/types';

export const DROP_SHADOW = 'drop-shadow(0px 0px 4px rgb(0, 0, 0)) drop-shadow(0px 0px 4px rgba(0, 0, 0, 0.3))';

export type ComparisonProps = {
  firstImage: ImageDTO;
  secondImage: ImageDTO;
  containerDims: Dimensions;
};

export const fitDimsToContainer = (containerDims: Dimensions, imageDims: Dimensions): Dimensions => {
  // Fall back to the image's dimensions if the container has no dimensions
  if (containerDims.width === 0 || containerDims.height === 0) {
    return { width: imageDims.width, height: imageDims.height };
  }

  // Fall back to the image's dimensions if the image fits within the container
  if (imageDims.width <= containerDims.width && imageDims.height <= containerDims.height) {
    return { width: imageDims.width, height: imageDims.height };
  }

  const targetAspectRatio = containerDims.width / containerDims.height;
  const imageAspectRatio = imageDims.width / imageDims.height;

  let width: number;
  let height: number;

  if (imageAspectRatio > targetAspectRatio) {
    // Image is wider than container's aspect ratio
    width = containerDims.width;
    height = width / imageAspectRatio;
  } else {
    // Image is taller than container's aspect ratio
    height = containerDims.height;
    width = height * imageAspectRatio;
  }
  return { width, height };
};

/**
 * Gets the dimensions of the second image in a comparison based on the comparison fit mode.
 */
export const getSecondImageDims = (
  comparisonFit: ComparisonFit,
  fittedDims: Dimensions,
  firstImageDims: Dimensions,
  secondImageDims: Dimensions
): Dimensions => {
  const width =
    comparisonFit === 'fill' ? fittedDims.width : (fittedDims.width * secondImageDims.width) / firstImageDims.width;
  const height =
    comparisonFit === 'fill' ? fittedDims.height : (fittedDims.height * secondImageDims.height) / firstImageDims.height;

  return { width, height };
};
export const selectComparisonImages = createMemoizedSelector(selectGallerySlice, (gallerySlice) => {
  const firstImage = gallerySlice.selection.slice(-1)[0] ?? null;
  const secondImage = gallerySlice.imageToCompare;
  return { firstImage, secondImage };
});
