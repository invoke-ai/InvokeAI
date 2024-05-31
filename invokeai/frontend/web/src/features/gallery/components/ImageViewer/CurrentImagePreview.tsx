import { Box, Flex } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { skipToken } from '@reduxjs/toolkit/query';
import { useAppSelector } from 'app/store/storeHooks';
import IAIDndImage from 'common/components/IAIDndImage';
import IAIDroppable from 'common/components/IAIDroppable';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import type { SelectForCompareDropData, TypesafeDraggableData } from 'features/dnd/types';
import ImageMetadataViewer from 'features/gallery/components/ImageMetadataViewer/ImageMetadataViewer';
import NextPrevImageButtons from 'features/gallery/components/NextPrevImageButtons';
import { selectLastSelectedImage } from 'features/gallery/store/gallerySelectors';
import type { AnimationProps } from 'framer-motion';
import { AnimatePresence, motion } from 'framer-motion';
import { memo, useCallback, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { PiImageBold } from 'react-icons/pi';
import { useGetImageDTOQuery } from 'services/api/endpoints/images';

import ProgressImage from './ProgressImage';

const selectLastSelectedImageName = createSelector(
  selectLastSelectedImage,
  (lastSelectedImage) => lastSelectedImage?.image_name
);

const droppableData: SelectForCompareDropData = {
  id: 'current-image',
  actionType: 'SELECT_FOR_COMPARE',
};

const CurrentImagePreview = () => {
  const { t } = useTranslation();
  const shouldShowImageDetails = useAppSelector((s) => s.ui.shouldShowImageDetails);
  const imageName = useAppSelector(selectLastSelectedImageName);
  const hasDenoiseProgress = useAppSelector((s) => Boolean(s.system.denoiseProgress));
  const shouldShowProgressInViewer = useAppSelector((s) => s.ui.shouldShowProgressInViewer);

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
      {hasDenoiseProgress && shouldShowProgressInViewer ? (
        <ProgressImage />
      ) : (
        <IAIDndImage
          imageDTO={imageDTO}
          draggableData={draggableData}
          isDropDisabled={true}
          isUploadDisabled={true}
          fitContainer
          useThumbailFallback
          dropLabel={t('gallery.setCurrentImage')}
          noContentFallback={<IAINoContentFallback icon={PiImageBold} label={t('gallery.noImageSelected')} />}
          dataTestId="image-preview"
        />
      )}
      <IAIDroppable data={droppableData} dropLabel="Select for Compare" />
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
            width="full"
            height="full"
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
