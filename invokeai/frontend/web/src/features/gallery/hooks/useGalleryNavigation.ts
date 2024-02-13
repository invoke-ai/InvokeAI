import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { getGalleryImageDataTestId } from 'features/gallery/components/ImageGrid/getGalleryImageDataTestId';
import { imageItemContainerTestId } from 'features/gallery/components/ImageGrid/ImageGridItemContainer';
import { imageListContainerTestId } from 'features/gallery/components/ImageGrid/ImageGridListContainer';
import { virtuosoGridRefs } from 'features/gallery/components/ImageGrid/types';
import { useGalleryImages } from 'features/gallery/hooks/useGalleryImages';
import { selectLastSelectedImage } from 'features/gallery/store/gallerySelectors';
import { imageSelected } from 'features/gallery/store/gallerySlice';
import { getIsVisible } from 'features/gallery/util/getIsVisible';
import { getScrollToIndexAlign } from 'features/gallery/util/getScrollToIndexAlign';
import { clamp } from 'lodash-es';
import { useCallback, useMemo } from 'react';
import type { ImageDTO } from 'services/api/types';
import { imagesSelectors } from 'services/api/util';

/**
 * This hook is used to navigate the gallery using the arrow keys.
 *
 * The gallery is rendered as a grid. In order to navigate the grid,
 * we need to know how many images are in each row and whether or not
 * an image is visible in the gallery.
 *
 * We use direct DOM query selectors to check if an image is visible
 * to avoid having to track a ref for each image.
 */

/**
 * Gets the number of images per row in the gallery by grabbing their DOM elements.
 */
const getImagesPerRow = (): number => {
  const widthOfGalleryImage =
    document.querySelector(`[data-testid="${imageItemContainerTestId}"]`)?.getBoundingClientRect().width ?? 1;

  const widthOfGalleryGrid =
    document.querySelector(`[data-testid="${imageListContainerTestId}"]`)?.getBoundingClientRect().width ?? 0;

  const imagesPerRow = Math.round(widthOfGalleryGrid / widthOfGalleryImage);

  return imagesPerRow;
};

/**
 * Scrolls to the image with the given name.
 * If the image is not fully visible, it will not be scrolled to.
 * @param imageName The image name to scroll to.
 * @param index The index of the image in the gallery.
 */
const scrollToImage = (imageName: string, index: number) => {
  const virtuosoContext = virtuosoGridRefs.get();
  const range = virtuosoContext.virtuosoRangeRef?.current;
  const root = virtuosoContext.rootRef?.current;
  const virtuoso = virtuosoContext.virtuosoRef?.current;

  if (!range || !virtuoso || !root) {
    return;
  }

  const imageElement = document.querySelector(`[data-testid="${getGalleryImageDataTestId(imageName)}"]`);
  const itemRect = imageElement?.getBoundingClientRect();
  const rootRect = root.getBoundingClientRect();
  if (!itemRect || !getIsVisible(itemRect, rootRect)) {
    virtuoso.scrollToIndex({
      index,
      align: getScrollToIndexAlign(index, range),
    });
  }
};

// Utilities to get the image to the left, right, up, or down of the current image.

const getLeftImage = (images: ImageDTO[], currentIndex: number) => {
  const index = clamp(currentIndex - 1, 0, images.length - 1);
  const image = images[index];
  return { index, image };
};

const getRightImage = (images: ImageDTO[], currentIndex: number) => {
  const index = clamp(currentIndex + 1, 0, images.length - 1);
  const image = images[index];
  return { index, image };
};

const getUpImage = (images: ImageDTO[], currentIndex: number) => {
  const imagesPerRow = getImagesPerRow();
  // If we are on the first row, we want to stay on the first row, not go to first image
  const isOnFirstRow = currentIndex < imagesPerRow;
  const index = isOnFirstRow ? currentIndex : clamp(currentIndex - imagesPerRow, 0, images.length - 1);
  const image = images[index];
  return { index, image };
};

const getDownImage = (images: ImageDTO[], currentIndex: number) => {
  const imagesPerRow = getImagesPerRow();
  // If there are no images below the current image, we want to stay where we are
  const areImagesBelow = currentIndex < images.length - imagesPerRow;
  const index = areImagesBelow ? clamp(currentIndex + imagesPerRow, 0, images.length - 1) : currentIndex;
  const image = images[index];
  return { index, image };
};

const getImageFuncs = {
  left: getLeftImage,
  right: getRightImage,
  up: getUpImage,
  down: getDownImage,
};

export type UseGalleryNavigationReturn = {
  handleLeftImage: () => void;
  handleRightImage: () => void;
  handleUpImage: () => void;
  handleDownImage: () => void;
  isOnFirstImage: boolean;
  isOnLastImage: boolean;
  areImagesBelowCurrent: boolean;
};

/**
 * Provides access to the gallery navigation via arrow keys.
 * Also provides information about the current image's position in the gallery,
 * useful for determining whether to load more images or display navigatin
 * buttons.
 */
export const useGalleryNavigation = (): UseGalleryNavigationReturn => {
  const dispatch = useAppDispatch();
  const lastSelectedImage = useAppSelector(selectLastSelectedImage);
  const {
    queryResult: { data },
  } = useGalleryImages();
  const loadedImagesCount = useMemo(() => data?.ids.length ?? 0, [data?.ids.length]);
  const lastSelectedImageIndex = useMemo(() => {
    if (!data || !lastSelectedImage) {
      return 0;
    }
    return imagesSelectors.selectAll(data).findIndex((i) => i.image_name === lastSelectedImage.image_name);
  }, [lastSelectedImage, data]);

  const handleNavigation = useCallback(
    (direction: 'left' | 'right' | 'up' | 'down') => {
      if (!data) {
        return;
      }
      const { index, image } = getImageFuncs[direction](imagesSelectors.selectAll(data), lastSelectedImageIndex);
      if (!image || index === lastSelectedImageIndex) {
        return;
      }
      dispatch(imageSelected(image));
      scrollToImage(image.image_name, index);
    },
    [dispatch, lastSelectedImageIndex, data]
  );

  const isOnFirstImage = useMemo(() => lastSelectedImageIndex === 0, [lastSelectedImageIndex]);

  const isOnLastImage = useMemo(
    () => lastSelectedImageIndex === loadedImagesCount - 1,
    [lastSelectedImageIndex, loadedImagesCount]
  );

  const areImagesBelowCurrent = useMemo(() => {
    const imagesPerRow = getImagesPerRow();
    return lastSelectedImageIndex + imagesPerRow < loadedImagesCount;
  }, [lastSelectedImageIndex, loadedImagesCount]);

  const handleLeftImage = useCallback(() => {
    handleNavigation('left');
  }, [handleNavigation]);

  const handleRightImage = useCallback(() => {
    handleNavigation('right');
  }, [handleNavigation]);

  const handleUpImage = useCallback(() => {
    handleNavigation('up');
  }, [handleNavigation]);

  const handleDownImage = useCallback(() => {
    handleNavigation('down');
  }, [handleNavigation]);

  return {
    handleLeftImage,
    handleRightImage,
    handleUpImage,
    handleDownImage,
    isOnFirstImage,
    isOnLastImage,
    areImagesBelowCurrent,
  };
};
