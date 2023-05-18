import { Box, Flex } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAICanvas from 'features/canvas/components/IAICanvas';
import IAICanvasResizer from 'features/canvas/components/IAICanvasResizer';
import IAICanvasToolbar from 'features/canvas/components/IAICanvasToolbar/IAICanvasToolbar';
import { canvasSelector } from 'features/canvas/store/canvasSelectors';
import { requestCanvasRescale } from 'features/canvas/store/thunks/requestCanvasScale';
import { isEqual } from 'lodash-es';

import { memo, useLayoutEffect } from 'react';

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

const UnifiedCanvasContent = () => {
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
        padding: 4,
        borderRadius: 'base',
        bg: 'base.850',
      }}
    >
      <Flex
        sx={{
          flexDirection: 'column',
          alignItems: 'center',
          gap: 4,
          width: '100%',
          height: '100%',
        }}
      >
        <IAICanvasToolbar />
        <Flex
          sx={{
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            gap: 4,
            width: '100%',
            height: '100%',
          }}
        >
          {doesCanvasNeedScaling ? <IAICanvasResizer /> : <IAICanvas />}
        </Flex>
      </Flex>
    </Box>
  );
};

export default memo(UnifiedCanvasContent);
