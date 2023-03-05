import { Box, BoxProps, Flex } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import ImageGallery from 'features/gallery/components/ImageGallery';
import { setInitialImage } from 'features/parameters/store/generationSlice';
import {
  activeTabNameSelector,
  uiSelector,
} from 'features/ui/store/uiSelectors';
import { DragEvent, ReactNode } from 'react';

import { setInitialCanvasImage } from 'features/canvas/store/canvasSlice';
import useGetImageByUuid from 'features/gallery/hooks/useGetImageByUuid';
import { lightboxSelector } from 'features/lightbox/store/lightboxSelectors';
import { isEqual } from 'lodash';

const workareaSelector = createSelector(
  [uiSelector, lightboxSelector, activeTabNameSelector],
  (ui, lightbox, activeTabName) => {
    const { shouldPinParametersPanel } = ui;
    const { isLightboxOpen } = lightbox;
    return {
      shouldPinParametersPanel,
      isLightboxOpen,
      activeTabName,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: isEqual,
    },
  }
);

type InvokeWorkareaProps = BoxProps & {
  optionsPanel: ReactNode;
  children: ReactNode;
};

const InvokeWorkarea = (props: InvokeWorkareaProps) => {
  const dispatch = useAppDispatch();
  const { optionsPanel, children, ...rest } = props;
  const { activeTabName, isLightboxOpen } = useAppSelector(workareaSelector);

  const getImageByUuid = useGetImageByUuid();

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    const uuid = e.dataTransfer.getData('invokeai/imageUuid');
    const image = getImageByUuid(uuid);
    if (!image) return;
    if (activeTabName === 'img2img') {
      dispatch(setInitialImage(image));
    } else if (activeTabName === 'unifiedCanvas') {
      dispatch(setInitialCanvasImage(image));
    }
  };

  return (
    <Box {...rest} pos="relative" w="100%" h="100%">
      <Flex gap={4} h="100%">
        {optionsPanel}
        <Box pos="relative" w="100%" h="100%" onDrop={handleDrop}>
          {children}
        </Box>
        {!isLightboxOpen && <ImageGallery />}
      </Flex>
    </Box>
  );
};

export default InvokeWorkarea;
