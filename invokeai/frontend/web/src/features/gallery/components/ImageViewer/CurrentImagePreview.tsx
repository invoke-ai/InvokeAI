import { Box, Flex } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { skipToken } from '@reduxjs/toolkit/query';
import { useAppSelector } from 'app/store/storeHooks';
import { CanvasAlertsInvocationProgress } from 'features/controlLayers/components/CanvasAlerts/CanvasAlertsInvocationProgress';
import { CanvasAlertsSendingToCanvas } from 'features/controlLayers/components/CanvasAlerts/CanvasAlertsSendingTo';
import { DndImage } from 'features/dnd/DndImage';
import ImageMetadataViewer from 'features/gallery/components/ImageMetadataViewer/ImageMetadataViewer';
import NextPrevImageButtons from 'features/gallery/components/NextPrevImageButtons';
import { selectLastSelectedImageName } from 'features/gallery/store/gallerySelectors';
import { selectShouldShowImageDetails, selectShouldShowProgressInViewer } from 'features/ui/store/uiSelectors';
import type { AnimationProps } from 'framer-motion';
import { AnimatePresence, motion } from 'framer-motion';
import { memo, useCallback, useRef, useState } from 'react';
import { useGetImageDTOQuery } from 'services/api/endpoints/images';
import type { ImageDTO } from 'services/api/types';
import { $hasProgressImage, $isProgressFromCanvas } from 'services/events/stores';

import { NoContentForViewer } from './NoContentForViewer';
import ProgressImage from './ProgressImage';

const CurrentImagePreview = () => {
  const shouldShowImageDetails = useAppSelector(selectShouldShowImageDetails);
  const imageName = useAppSelector(selectLastSelectedImageName);

  const { currentData: imageDTO } = useGetImageDTOQuery(imageName ?? skipToken);

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
      <ImageContent imageDTO={imageDTO} />
      <Flex
        flexDir="column"
        gap={2}
        position="absolute"
        top={0}
        insetInlineStart={0}
        pointerEvents="none"
        alignItems="flex-start"
      >
        <CanvasAlertsSendingToCanvas />
        <CanvasAlertsInvocationProgress />
      </Flex>
      {shouldShowImageDetails && imageDTO && (
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
            <NextPrevImageButtons />
          </Box>
        )}
      </AnimatePresence>
    </Flex>
  );
};

export default memo(CurrentImagePreview);

const ImageContent = memo(({ imageDTO }: { imageDTO?: ImageDTO }) => {
  const hasProgressImage = useStore($hasProgressImage);
  const isProgressFromCanvas = useStore($isProgressFromCanvas);
  const shouldShowProgressInViewer = useAppSelector(selectShouldShowProgressInViewer);

  if (hasProgressImage && !isProgressFromCanvas && shouldShowProgressInViewer) {
    return <ProgressImage />;
  }

  if (!imageDTO) {
    return <NoContentForViewer />;
  }

  return (
    <Flex w="full" h="full" position="absolute" alignItems="center" justifyContent="center">
      <DndImage imageDTO={imageDTO} />
    </Flex>
  );
});
ImageContent.displayName = 'ImageContent';

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
