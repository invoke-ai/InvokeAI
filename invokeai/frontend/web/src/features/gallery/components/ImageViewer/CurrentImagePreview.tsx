import { Box, Flex } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { skipToken } from '@reduxjs/toolkit/query';
import { useAppSelector } from 'app/store/storeHooks';
import IAIDndImage from 'common/components/IAIDndImage';
import type { TypesafeDraggableData } from 'features/dnd/types';
import ImageMetadataViewer from 'features/gallery/components/ImageMetadataViewer/ImageMetadataViewer';
import NextPrevImageButtons from 'features/gallery/components/NextPrevImageButtons';
import { selectLastSelectedImageName } from 'features/gallery/store/gallerySelectors';
import { selectShouldShowImageDetails, selectShouldShowProgressInViewer } from 'features/ui/store/uiSelectors';
import type { AnimationProps } from 'framer-motion';
import { AnimatePresence, motion } from 'framer-motion';
import { memo, useCallback, useMemo, useRef, useState } from 'react';
import { useGetImageDTOQuery } from 'services/api/endpoints/images';
import { $hasProgressImage, $isProgressFromCanvas } from 'services/events/stores';

import { NoContentForViewer } from './NoContentForViewer';
import ProgressImage from './ProgressImage';

const CurrentImagePreview = () => {
  const shouldShowImageDetails = useAppSelector(selectShouldShowImageDetails);
  const imageName = useAppSelector(selectLastSelectedImageName);
  const hasProgressImage = useStore($hasProgressImage);
  const isProgressFromCanvas = useStore($isProgressFromCanvas);
  const shouldShowProgressInViewer = useAppSelector(selectShouldShowProgressInViewer);

  const { currentData: imageDTO } = useGetImageDTOQuery(imageName ?? skipToken);

  const draggableData = useMemo<TypesafeDraggableData | undefined>(() => {
    if (imageDTO) {
      return {
        id: 'current-image',
        payloadType: 'IMAGE_DTO',
        payload: { imageDTO },
      };
    }
  }, [imageDTO]);

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
      {hasProgressImage && !isProgressFromCanvas && shouldShowProgressInViewer ? (
        <ProgressImage />
      ) : (
        <IAIDndImage
          imageDTO={imageDTO}
          draggableData={draggableData}
          isDropDisabled={true}
          isUploadDisabled={true}
          fitContainer
          useThumbailFallback
          noContentFallback={<NoContentForViewer />}
          dataTestId="image-preview"
        />
      )}
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
