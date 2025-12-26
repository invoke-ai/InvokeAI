import { Box, Flex } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppSelector } from 'app/store/storeHooks';
import { CanvasAlertsInvocationProgress } from 'features/controlLayers/components/CanvasAlerts/CanvasAlertsInvocationProgress';
import { DndImage } from 'features/dnd/DndImage';
import ImageMetadataViewer from 'features/gallery/components/ImageMetadataViewer/ImageMetadataViewer';
import NextPrevItemButtons from 'features/gallery/components/NextPrevItemButtons';
import { selectShouldShowItemDetails, selectShouldShowProgressInViewer } from 'features/ui/store/uiSelectors';
import type { AnimationProps } from 'framer-motion';
import { AnimatePresence, motion, useMotionValue } from 'framer-motion';
import { memo, useCallback, useEffect, useRef, useState } from 'react';
import { useImageDTO } from 'services/api/endpoints/images';
import type { ImageDTO } from 'services/api/types';

import { useImageViewerContext } from './context';
import { NoContentForViewer } from './NoContentForViewer';
import { ProgressImage } from './ProgressImage2';
import { ProgressIndicator } from './ProgressIndicator2';
import { useSwipeNavigation } from './useSwipeNavigation';

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

  // Track drag state for showing adjacent images
  const dragX = useMotionValue(0);
  const [isDragging, setIsDragging] = useState(false);

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
    dragX.set(0);
    setIsDragging(false);
  }, [imageDTO?.image_name, dragX]);

  const onDragStart = useCallback(() => {
    setIsDragging(true);
  }, []);

  const handleDragEnd = useCallback(
    (event: MouseEvent | TouchEvent | PointerEvent, info: { offset: { x: number; y: number } }) => {
      setIsDragging(false);
      onDragEnd(event, info);
    },
    [onDragEnd]
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
        dragConstraints={{ left: 0, right: 0 }}
        dragElastic={0.2}
        onDragStart={onDragStart}
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
          {isDragging && previousImageDTO && (
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
          {isDragging && nextImageDTO && (
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
