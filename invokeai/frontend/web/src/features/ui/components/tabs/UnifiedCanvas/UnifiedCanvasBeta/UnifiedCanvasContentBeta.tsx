import { createSelector } from '@reduxjs/toolkit';
// import IAICanvas from 'features/canvas/components/IAICanvas';
import { Box, Flex } from '@chakra-ui/react';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAICanvas from 'features/canvas/components/IAICanvas';
import IAICanvasResizer from 'features/canvas/components/IAICanvasResizer';
import { canvasSelector } from 'features/canvas/store/canvasSelectors';

import { isEqual } from 'lodash-es';
import { useLayoutEffect } from 'react';
import UnifiedCanvasToolbarBeta from './UnifiedCanvasToolbarBeta';
import UnifiedCanvasToolSettingsBeta from './UnifiedCanvasToolSettingsBeta';
import { requestCanvasRescale } from 'features/canvas/store/thunks/requestCanvasScale';

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

const UnifiedCanvasContentBeta = () => {
  const dispatch = useAppDispatch();

  const { doesCanvasNeedScaling } = useAppSelector(selector);

  useLayoutEffect(() => {
    dispatch(requestCanvasRescale());
    const resizeCallback = () => {
      dispatch(requestCanvasRescale());
    };

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

export default UnifiedCanvasContentBeta;
