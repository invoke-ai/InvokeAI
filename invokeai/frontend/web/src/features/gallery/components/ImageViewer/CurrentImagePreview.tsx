import { Box, Flex } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppSelector } from 'app/store/storeHooks';
import { CanvasAlertsInvocationProgress } from 'features/controlLayers/components/CanvasAlerts/CanvasAlertsInvocationProgress';
import { DndImage } from 'features/dnd/DndImage';
import ImageMetadataViewer from 'features/gallery/components/ImageMetadataViewer/ImageMetadataViewer';
import NextPrevItemButtons from 'features/gallery/components/NextPrevItemButtons';
import { useNextPrevItemNavigation } from 'features/gallery/components/useNextPrevItemNavigation';
import { selectLastSelectedItem } from 'features/gallery/store/gallerySelectors';
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import { navigationApi } from 'features/ui/layouts/navigation-api';
import {
  selectActiveTab,
  selectShouldShowItemDetails,
  selectShouldShowProgressInViewer,
} from 'features/ui/store/uiSelectors';
import type { AnimationProps } from 'framer-motion';
import { AnimatePresence, motion } from 'framer-motion';
import { memo, useCallback, useEffect, useRef, useState } from 'react';
import type { ImageDTO } from 'services/api/types';

import { useImageViewerContext } from './context';
import { NoContentForViewer } from './NoContentForViewer';
import { ProgressImage } from './ProgressImage2';
import { ProgressImageTiles } from './ProgressImageTiles';
import { ProgressIndicator } from './ProgressIndicator2';

export const CurrentImagePreview = memo(({ imageDTO }: { imageDTO: ImageDTO | null }) => {
  const activeTab = useAppSelector(selectActiveTab);
  const selectedImageName = useAppSelector(selectLastSelectedItem);
  const shouldShowItemDetails = useAppSelector(selectShouldShowItemDetails);
  const shouldShowProgressInViewer = useAppSelector(selectShouldShowProgressInViewer);
  const { goToPreviousImage, goToNextImage, isFetching } = useNextPrevItemNavigation();
  const {
    onLoadImage,
    $progressEvent,
    $progressImage,
    $activeProgressData,
    $isProgressImageResolving,
    $isTemporarilyShowingSelectedImage,
  } = useImageViewerContext();
  const progressEvent = useStore($progressEvent);
  const progressImage = useStore($progressImage);
  const activeProgressData = useStore($activeProgressData);
  const isProgressImageResolving = useStore($isProgressImageResolving);
  const isTemporarilyShowingSelectedImage = useStore($isTemporarilyShowingSelectedImage);
  const [imageToRender, setImageToRender] = useState<ImageDTO | null>(null);
  const previousRenderedImageNameRef = useRef<string | null>(null);
  const selectedImageRevealTimeoutId = useRef(0);

  useEffect(() => {
    if (!selectedImageName) {
      setImageToRender(null);
      return;
    }

    if (!imageDTO || imageToRender?.image_name === imageDTO.image_name) {
      return;
    }

    let canceled = false;

    const onReady = () => {
      if (canceled) {
        return;
      }
      setImageToRender(imageDTO);
    };

    if (typeof window === 'undefined') {
      onReady();
      return;
    }

    const preloader = new window.Image();

    preloader.onload = onReady;
    preloader.onerror = onReady;
    preloader.src = imageDTO.image_url;

    if (preloader.complete) {
      onReady();
    }

    return () => {
      canceled = true;
      preloader.onload = null;
      preloader.onerror = null;
    };
  }, [imageDTO, imageToRender?.image_name, selectedImageName]);

  const hasProgressImage = progressImage !== null;

  useEffect(() => {
    const renderedImageName = imageToRender?.image_name ?? null;
    const previousRenderedImageName = previousRenderedImageNameRef.current;
    previousRenderedImageNameRef.current = renderedImageName;

    window.clearTimeout(selectedImageRevealTimeoutId.current);

    if (
      !shouldShowProgressInViewer ||
      !hasProgressImage ||
      isProgressImageResolving ||
      !renderedImageName ||
      renderedImageName !== selectedImageName
    ) {
      $isTemporarilyShowingSelectedImage.set(false);
      return;
    }

    if (previousRenderedImageName === null || previousRenderedImageName === renderedImageName) {
      return;
    }

    $isTemporarilyShowingSelectedImage.set(true);
    selectedImageRevealTimeoutId.current = window.setTimeout(() => {
      $isTemporarilyShowingSelectedImage.set(false);
    }, SELECTED_IMAGE_REVEAL_DURATION_MS);

    return () => {
      window.clearTimeout(selectedImageRevealTimeoutId.current);
    };
  }, [
    $isTemporarilyShowingSelectedImage,
    hasProgressImage,
    imageToRender?.image_name,
    isProgressImageResolving,
    selectedImageName,
    shouldShowProgressInViewer,
  ]);

  useEffect(() => {
    return () => {
      $isTemporarilyShowingSelectedImage.set(false);
    };
  }, [$isTemporarilyShowingSelectedImage]);

  // Show and hide the next/prev buttons on mouse move
  const [shouldShowNextPrevButtons, setShouldShowNextPrevButtons] = useState<boolean>(false);
  const timeoutId = useRef(0);
  const onMouseOver = useCallback(() => {
    setShouldShowNextPrevButtons(true);
    window.clearTimeout(timeoutId.current);
  }, []);
  const onMouseOut = useCallback(() => {
    timeoutId.current = window.setTimeout(() => {
      setShouldShowNextPrevButtons(false);
    }, 500);
  }, []);

  const handleViewerArrowNavigation = useCallback(
    (event: KeyboardEvent, navigate: () => void) => {
      if (!navigationApi.isViewerArrowNavigationMode(activeTab) || !imageToRender || isFetching) {
        return;
      }
      if (event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement) {
        return;
      }
      event.preventDefault();
      navigate();
    },
    [activeTab, imageToRender, isFetching]
  );

  const onHotkeyPrevImage = useCallback(
    (event: KeyboardEvent) => {
      handleViewerArrowNavigation(event, goToPreviousImage);
    },
    [goToPreviousImage, handleViewerArrowNavigation]
  );

  const onHotkeyNextImage = useCallback(
    (event: KeyboardEvent) => {
      handleViewerArrowNavigation(event, goToNextImage);
    },
    [goToNextImage, handleViewerArrowNavigation]
  );

  useRegisteredHotkeys({
    id: 'galleryNavLeft',
    category: 'gallery',
    callback: onHotkeyPrevImage,
    options: { preventDefault: true },
    dependencies: [onHotkeyPrevImage],
  });

  useRegisteredHotkeys({
    id: 'galleryNavRight',
    category: 'gallery',
    callback: onHotkeyNextImage,
    options: { preventDefault: true },
    dependencies: [onHotkeyNextImage],
  });

  const withProgress = shouldShowProgressInViewer && hasProgressImage && !isTemporarilyShowingSelectedImage;
  // When more than one session is generating concurrently (multi-GPU), tile their previews instead of
  // showing only the most recent one.
  const withTiledProgress = withProgress && activeProgressData.length > 1;

  return (
    <Flex
      onMouseOver={onMouseOver}
      onMouseOut={onMouseOut}
      width="full"
      height="full"
      alignItems="center"
      justifyContent="center"
      position="relative"
    >
      {imageToRender && (
        <Flex w="full" h="full" position="absolute" alignItems="center" justifyContent="center">
          <DndImage imageDTO={imageToRender} onLoad={onLoadImage} borderRadius="base" />
        </Flex>
      )}
      {!imageToRender && <NoContentForViewer />}
      {withProgress && (
        <Flex w="full" h="full" position="absolute" alignItems="center" justifyContent="center" bg="base.900">
          {withTiledProgress ? (
            <ProgressImageTiles data={activeProgressData} />
          ) : (
            <>
              <ProgressImage progressImage={progressImage} />
              {progressEvent && (
                <ProgressIndicator progressEvent={progressEvent} position="absolute" top={6} right={6} size={8} />
              )}
            </>
          )}
        </Flex>
      )}
      <Flex flexDir="column" gap={2} position="absolute" top={0} insetInlineStart={0} alignItems="flex-start">
        <CanvasAlertsInvocationProgress />
      </Flex>
      {shouldShowItemDetails && imageToRender && !withProgress && (
        <Box position="absolute" opacity={0.8} top={0} width="full" height="full" borderRadius="base">
          <ImageMetadataViewer image={imageToRender} />
        </Box>
      )}
      <AnimatePresence>
        {shouldShowNextPrevButtons && imageToRender && (
          <Box
            as={motion.div}
            key="nextPrevButtons"
            initial={initial}
            animate={animateArrows}
            exit={exit}
            position="absolute"
            top={0}
            right={0}
            bottom={0}
            left={0}
            pointerEvents="none"
          >
            <NextPrevItemButtons />
          </Box>
        )}
      </AnimatePresence>
    </Flex>
  );
});
CurrentImagePreview.displayName = 'CurrentImagePreview';

const initial: AnimationProps['initial'] = {
  opacity: 0,
};
const animateArrows: AnimationProps['animate'] = {
  opacity: 1,
  transition: { duration: 0.07 },
};
const exit: AnimationProps['exit'] = {
  opacity: 0,
  transition: { duration: 0.07 },
};

const SELECTED_IMAGE_REVEAL_DURATION_MS = 2000;
