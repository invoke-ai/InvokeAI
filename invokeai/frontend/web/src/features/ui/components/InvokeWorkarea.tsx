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
    <Flex {...rest} pos="relative" w="full" h={APP_CONTENT_HEIGHT} gap={4}>
      <ParametersPanel>{parametersPanelContent}</ParametersPanel>
      <Box pos="relative" w="100%" h="100%" onDrop={handleDrop}>
        {children}
      </Box>
    </Flex>
  );
};

export default InvokeWorkarea;
