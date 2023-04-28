import { Box, BoxProps, Grid, GridItem } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { initialImageSelected } from 'features/parameters/store/generationSlice';
import {
  activeTabNameSelector,
  uiSelector,
} from 'features/ui/store/uiSelectors';
import { DragEvent, ReactNode } from 'react';

import { setInitialCanvasImage } from 'features/canvas/store/canvasSlice';
import useGetImageByUuid from 'features/gallery/hooks/useGetImageByUuid';
import { isEqual } from 'lodash-es';
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
    if (activeTabName === 'unifiedCanvas') {
      dispatch(setInitialCanvasImage(image));
    }
  };

  return (
    <Grid
      {...rest}
      gridTemplateAreas={{
        base: `'workarea-display' 'workarea-panel'`,
        xl: `'workarea-panel workarea-display'`,
      }}
      gridAutoRows={{ base: 'auto', xl: 'auto' }}
      gridAutoColumns={{ md: 'max-content auto' }}
      pos="relative"
      w="full"
      h={{ base: 'auto', xl: APP_CONTENT_HEIGHT }}
      gap={4}
    >
      <ParametersPanel>{parametersPanelContent}</ParametersPanel>
      <GridItem gridArea="workarea-display">
        <Box
          pos="relative"
          w={{ base: '100vw', xl: 'full' }}
          paddingRight={{ base: 8, xl: 0 }}
          h={{ base: 600, xl: '100%' }}
          onDrop={handleDrop}
        >
          {children}
        </Box>
      </GridItem>
    </Grid>
  );
};

export default InvokeWorkarea;
