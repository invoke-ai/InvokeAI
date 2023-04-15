import { Box, BoxProps, Flex } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import { setInitialImage } from 'features/parameters/store/generationSlice';
import {
  activeTabNameSelector,
  uiSelector,
} from 'features/ui/store/uiSelectors';
import { DragEvent, ReactNode } from 'react';

import { setInitialCanvasImage } from 'features/canvas/store/canvasSlice';
import useGetImageByUuid from 'features/gallery/hooks/useGetImageByUuid';
import { isEqual } from 'lodash';
import { APP_CONTENT_HEIGHT } from 'theme/util/constants';
import ParametersPanel from './ParametersPanel';
import MediaQuery from 'react-responsive';
import ImageGalleryPanel from 'features/gallery/components/ImageGalleryPanel';
import { isMobile } from 'theme/util/isMobile';

const workareaSelector = createSelector(
  [uiSelector, activeTabNameSelector],
  (ui, activeTabName) => {
    const { shouldPinParametersPanel } = ui;
    return {
      shouldPinParametersPanel,
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
  parametersPanelContent: ReactNode;
  children: ReactNode;
};

const InvokeWorkarea = (props: InvokeWorkareaProps) => {
  const { parametersPanelContent, children, ...rest } = props;
  const dispatch = useAppDispatch();
  const { activeTabName } = useAppSelector(workareaSelector);

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
    <>
      <MediaQuery minDeviceWidth={768}>
        <Flex {...rest} pos="relative" w="full" h={APP_CONTENT_HEIGHT} gap={4}>
          <ParametersPanel>{parametersPanelContent}</ParametersPanel>
          <Box pos="relative" w="100%" h="100%" onDrop={handleDrop}>
            {children}
          </Box>
        </Flex>
      </MediaQuery>
      <MediaQuery maxDeviceWidth={768}>
        <Flex
          {...rest}
          display="block"
          pos="relative"
          w="full"
          h={APP_CONTENT_HEIGHT}
        >
          <Box pos="sticky" w="full" h="70%" onDrop={handleDrop}>
            {children}
          </Box>
          <ParametersPanel>{parametersPanelContent}</ParametersPanel>
          {isMobile && <ImageGalleryPanel />}
        </Flex>
      </MediaQuery>
    </>
  );
};

export default InvokeWorkarea;
