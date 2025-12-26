import { Box, Flex } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppSelector } from 'app/store/storeHooks';
import { CanvasAlertsInvocationProgress } from 'features/controlLayers/components/CanvasAlerts/CanvasAlertsInvocationProgress';
import { DndImage } from 'features/dnd/DndImage';
import ImageMetadataViewer from 'features/gallery/components/ImageMetadataViewer/ImageMetadataViewer';
import NextPrevItemButtons from 'features/gallery/components/NextPrevItemButtons';
import { selectShouldShowItemDetails, selectShouldShowProgressInViewer } from 'features/ui/store/uiSelectors';
import type { AnimationProps } from 'framer-motion';
import { animate, AnimatePresence, motion, useMotionValue } from 'framer-motion';
import { memo, useCallback, useEffect, useRef, useState } from 'react';
import { useImageDTO } from 'services/api/endpoints/images';
import type { ImageDTO } from 'services/api/types';

import { useImageViewerContext } from './context';
import { NoContentForViewer } from './NoContentForViewer';
import { ProgressImage } from './ProgressImage2';
import { ProgressIndicator } from './ProgressIndicator2';
import { useSwipeNavigation } from './useSwipeNavigation';

// Minimum duration for carousel transitions (in seconds)
// This prevents instant transitions when users flick rapidly
const MIN_TRANSITION_DURATION = 0.3;

export const CurrentImagePreview = memo(({ imageDTO }: { imageDTO: ImageDTO | null }) => {
  const shouldShowItemDetails = useAppSelector(selectShouldShowItemDetails);
  const shouldShowProgressInViewer = useAppSelector(selectShouldShowProgressInViewer);
  const { onLoadImage, $progressEvent, $progressImage } = useImageViewerContext();
  const progressEvent = useStore($progressEvent);
  const progressImage = useStore($progressImage);
  const { onDragEnd, previousImageName, nextImageName } = useSwipeNavigation();

  // Get adjacent images for swipe preview
  const previousImageDTO = useImageDTO(previousImageName);
  const nextImageDTO = useImageDTO(nextImageName);

  // Track drag state
  const dragX = useMotionValue(0);

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

  const withProgress = shouldShowProgressInViewer && progressImage !== null;

  // Reset drag state when image changes
  useEffect(() => {
    // Instantly reset position when image changes (navigation occurred)
    dragX.set(0);
  }, [imageDTO?.image_name, dragX]);

  const handleDragEnd = useCallback(
    (event: MouseEvent | TouchEvent | PointerEvent, info: { offset: { x: number; y: number } }) => {
      // Check if navigation will occur (threshold is 50px in useSwipeNavigation)
      const willNavigate = Math.abs(info.offset.x) > 50;

      onDragEnd(event, info);

      // Only animate back to center if navigation did NOT occur
      // If navigation occurs, the useEffect will reset position when image changes
      if (!willNavigate) {
        animate(dragX, 0, {
          type: 'spring',
          stiffness: 300,
          damping: 30,
          duration: MIN_TRANSITION_DURATION,
        });
      }
    },
    [onDragEnd, dragX]
  );

  return (
    <Box
      onMouseOver={onMouseOver}
      onMouseOut={onMouseOut}
      width="full"
      height="full"
      position="relative"
      overflow="hidden"
    >
      {/* Horizontal carousel container with 3 image slots */}
      <Box
        as={motion.div}
        drag={imageDTO ? 'x' : false}
        dragElastic={0.1}
        dragMomentum={false}
        onDragEnd={handleDragEnd}
        style={{ x: dragX }}
        width="300%"
        height="full"
        display="flex"
        flexDirection="row"
        position="absolute"
        left="-100%"
      >
        {/* Previous image slot (left third) */}
        <Box
          width="33.333%"
          height="full"
          display="flex"
          alignItems="center"
          justifyContent="center"
          position="relative"
          flexShrink={0}
        >
          {previousImageDTO && (
            <Box
              position="absolute"
              top={0}
              left={0}
              right={0}
              bottom={0}
              display="flex"
              alignItems="center"
              justifyContent="center"
              pointerEvents="none"
            >
              <DndImage imageDTO={previousImageDTO} borderRadius="base" />
            </Box>
          )}
        </Box>

        {/* Current image slot (middle third) */}
        <Box
          width="33.333%"
          height="full"
          display="flex"
          alignItems="center"
          justifyContent="center"
          position="relative"
          flexShrink={0}
        >
          {imageDTO && (
            <Box
              position="absolute"
              top={0}
              left={0}
              right={0}
              bottom={0}
              display="flex"
              alignItems="center"
              justifyContent="center"
            >
              <DndImage imageDTO={imageDTO} onLoad={onLoadImage} borderRadius="base" />
            </Box>
          )}
        </Box>

        {/* Next image slot (right third) */}
        <Box
          width="33.333%"
          height="full"
          display="flex"
          alignItems="center"
          justifyContent="center"
          position="relative"
          flexShrink={0}
        >
          {nextImageDTO && (
            <Box
              position="absolute"
              top={0}
              left={0}
              right={0}
              bottom={0}
              display="flex"
              alignItems="center"
              justifyContent="center"
              pointerEvents="none"
            >
              <DndImage imageDTO={nextImageDTO} borderRadius="base" />
            </Box>
          )}
        </Box>
      </Box>

      {!imageDTO && <NoContentForViewer />}
      {withProgress && (
        <Flex w="full" h="full" position="absolute" alignItems="center" justifyContent="center" bg="base.900">
          <ProgressImage progressImage={progressImage} />
          {progressEvent && (
            <ProgressIndicator progressEvent={progressEvent} position="absolute" top={6} right={6} size={8} />
          )}
        </Flex>
      )}
      <Flex flexDir="column" gap={2} position="absolute" top={0} insetInlineStart={0} alignItems="flex-start">
        <CanvasAlertsInvocationProgress />
      </Flex>
      {shouldShowItemDetails && imageDTO && !withProgress && (
        <Box position="absolute" opacity={0.8} top={0} width="full" height="full" borderRadius="base">
          <ImageMetadataViewer image={imageDTO} />
        </Box>
      )}
      <AnimatePresence>
        {shouldShowNextPrevButtons && imageDTO && (
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
    </Box>
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
