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

import {
  setDoesCanvasNeedScaling,
  setInitialCanvasImage,
} from 'features/canvas/store/canvasSlice';
import useGetImageByUuid from 'features/gallery/hooks/useGetImageByUuid';
import { lightboxSelector } from 'features/lightbox/store/lightboxSelectors';
import { isEqual } from 'lodash';
import {
  APP_CONTENT_HEIGHT,
  PARAMETERS_PANEL_WIDTH,
} from 'theme/util/constants';
import ResizableDrawer from 'features/ui/components/common/ResizableDrawer/ResizableDrawer';
import {
  setShouldPinParametersPanel,
  setShouldShowParametersPanel,
} from 'features/ui/store/uiSlice';
import { useHotkeys } from 'react-hotkeys-hook';
import InvokeAILogoComponent from 'features/system/components/InvokeAILogoComponent';

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
  parametersPanel: ReactNode;
  children: ReactNode;
};

const InvokeWorkarea = (props: InvokeWorkareaProps) => {
  const { parametersPanel, children, ...rest } = props;
  const dispatch = useAppDispatch();
  const { activeTabName, isLightboxOpen } = useAppSelector(workareaSelector);
  const { shouldPinParametersPanel, shouldShowParametersPanel } =
    useAppSelector(uiSelector);

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

  const closeParametersPanel = () => {
    dispatch(setShouldShowParametersPanel(false));
  };

  useHotkeys(
    'o',
    () => {
      dispatch(setShouldShowParametersPanel(!shouldShowParametersPanel));
      shouldPinParametersPanel &&
        setTimeout(() => dispatch(setDoesCanvasNeedScaling(true)), 400);
    },
    [shouldShowParametersPanel, shouldPinParametersPanel]
  );

  useHotkeys(
    'esc',
    () => {
      dispatch(setShouldShowParametersPanel(false));
    },
    {
      enabled: () => !shouldPinParametersPanel,
      preventDefault: true,
    },
    [shouldPinParametersPanel]
  );

  useHotkeys(
    'shift+o',
    () => {
      dispatch(setShouldPinParametersPanel(!shouldPinParametersPanel));
      dispatch(setDoesCanvasNeedScaling(true));
    },
    [shouldPinParametersPanel]
  );

  return (
    <Flex {...rest} pos="relative" h={APP_CONTENT_HEIGHT} gap={4}>
      <ResizableDrawer
        direction="left"
        isResizable={true}
        shouldAllowResize={!shouldPinParametersPanel}
        isOpen={shouldShowParametersPanel || shouldPinParametersPanel}
        onClose={closeParametersPanel}
        isPinned={shouldPinParametersPanel}
        handleWidth={5}
        handleInteractWidth={'15px'}
        sx={{
          borderColor: 'base.700',
          p: shouldPinParametersPanel ? 0 : 4,
          bg: 'base.900',
        }}
        initialWidth={PARAMETERS_PANEL_WIDTH}
        minWidth={PARAMETERS_PANEL_WIDTH}
        pinnedWidth={PARAMETERS_PANEL_WIDTH}
        pinnedHeight={APP_CONTENT_HEIGHT}
      >
        <Flex
          flexDir="column"
          rowGap={4}
          paddingTop={!shouldPinParametersPanel ? 1.5 : 0}
        >
          {!shouldPinParametersPanel && <InvokeAILogoComponent />}
          {parametersPanel}
        </Flex>
      </ResizableDrawer>
      <Box pos="relative" w="100%" h="100%" onDrop={handleDrop}>
        {children}
      </Box>
      {!isLightboxOpen && <ImageGallery />}
    </Flex>
  );
};

export default InvokeWorkarea;
