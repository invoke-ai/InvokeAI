import { createSelector } from '@reduxjs/toolkit';
// import IAICanvas from 'features/canvas/components/IAICanvas';
import { Box, Flex } from '@chakra-ui/react';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import IAICanvas from 'features/canvas/components/IAICanvas';
import IAICanvasResizer from 'features/canvas/components/IAICanvasResizer';
import { canvasSelector } from 'features/canvas/store/canvasSelectors';
import { setDoesCanvasNeedScaling } from 'features/canvas/store/canvasSlice';

import { debounce, isEqual } from 'lodash';
import { useLayoutEffect } from 'react';
import UnifiedCanvasToolbarBeta from './UnifiedCanvasToolbarBeta';
import UnifiedCanvasToolSettingsBeta from './UnifiedCanvasToolSettingsBeta';

const selector = createSelector(
  [canvasSelector],
  (canvas) => {
    const { doesCanvasNeedScaling } = canvas;
    return {
      doesCanvasNeedScaling,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: isEqual,
    },
  }
);

const UnifiedCanvasDisplayBeta = () => {
  const dispatch = useAppDispatch();

  const { doesCanvasNeedScaling } = useAppSelector(selector);

  useLayoutEffect(() => {
    dispatch(setDoesCanvasNeedScaling(true));

    const resizeCallback = debounce(() => {
      dispatch(setDoesCanvasNeedScaling(true));
    }, 250);

    window.addEventListener('resize', resizeCallback);

    return () => window.removeEventListener('resize', resizeCallback);
  }, [dispatch]);

  return (
    <Box
      sx={{
        width: '100%',
        height: '100%',
        borderRadius: 'base',
        bg: 'base.850',
      }}
    >
      <Flex
        flexDirection="row"
        width="100%"
        height="100%"
        columnGap={4}
        padding={4}
      >
        <UnifiedCanvasToolbarBeta />
        <Flex width="100%" height="100%" flexDirection="column" rowGap={4}>
          <UnifiedCanvasToolSettingsBeta />
          {doesCanvasNeedScaling ? <IAICanvasResizer /> : <IAICanvas />}
        </Flex>
      </Flex>
    </Box>
  );
};

export default UnifiedCanvasDisplayBeta;
