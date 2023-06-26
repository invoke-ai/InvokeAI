import { Box, Flex } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAICanvas from 'features/canvas/components/IAICanvas';
import IAICanvasResizer from 'features/canvas/components/IAICanvasResizer';
import IAICanvasToolbar from 'features/canvas/components/IAICanvasToolbar/IAICanvasToolbar';
import { canvasSelector } from 'features/canvas/store/canvasSelectors';
import { requestCanvasRescale } from 'features/canvas/store/thunks/requestCanvasScale';
import { uiSelector } from 'features/ui/store/uiSelectors';

import { memo, useCallback, useLayoutEffect } from 'react';
import UnifiedCanvasToolbarBeta from './UnifiedCanvasBeta/UnifiedCanvasToolbarBeta';
import UnifiedCanvasToolSettingsBeta from './UnifiedCanvasBeta/UnifiedCanvasToolSettingsBeta';
import { ImageDTO } from 'services/api/types';
import { setInitialCanvasImage } from 'features/canvas/store/canvasSlice';
import { useDroppable } from '@dnd-kit/core';
import IAIDropOverlay from 'common/components/IAIDropOverlay';

const selector = createSelector(
  [canvasSelector, uiSelector],
  (canvas, ui) => {
    const { doesCanvasNeedScaling } = canvas;
    const { shouldUseCanvasBetaLayout } = ui;
    return {
      doesCanvasNeedScaling,
      shouldUseCanvasBetaLayout,
    };
  },
  defaultSelectorOptions
);

const UnifiedCanvasContent = () => {
  const dispatch = useAppDispatch();

  const { doesCanvasNeedScaling, shouldUseCanvasBetaLayout } =
    useAppSelector(selector);

  const onDrop = useCallback(
    (droppedImage: ImageDTO) => {
      dispatch(setInitialCanvasImage(droppedImage));
    },
    [dispatch]
  );

  const {
    isOver,
    setNodeRef: setDroppableRef,
    active,
  } = useDroppable({
    id: 'unifiedCanvas',
    data: {
      handleDrop: onDrop,
    },
  });

  useLayoutEffect(() => {
    const resizeCallback = () => {
      dispatch(requestCanvasRescale());
    };

    window.addEventListener('resize', resizeCallback);

    return () => window.removeEventListener('resize', resizeCallback);
  }, [dispatch]);

  if (shouldUseCanvasBetaLayout) {
    return (
      <Box
        ref={setDroppableRef}
        tabIndex={0}
        sx={{
          w: 'full',
          h: 'full',
          borderRadius: 'base',
          bg: 'base.850',
          p: 4,
        }}
      >
        <Flex
          sx={{
            w: 'full',
            h: 'full',
            gap: 4,
          }}
        >
          <UnifiedCanvasToolbarBeta />
          <Flex
            sx={{
              flexDir: 'column',
              w: 'full',
              h: 'full',
              gap: 4,
              position: 'relative',
            }}
          >
            <UnifiedCanvasToolSettingsBeta />
            <Box sx={{ w: 'full', h: 'full', position: 'relative' }}>
              {doesCanvasNeedScaling ? <IAICanvasResizer /> : <IAICanvas />}
              {active && <IAIDropOverlay isOver={isOver} />}
            </Box>
          </Flex>
        </Flex>
      </Box>
    );
  }

  return (
    <Box
      ref={setDroppableRef}
      tabIndex={-1}
      sx={{
        w: 'full',
        h: 'full',
        borderRadius: 'base',
        bg: 'base.850',
        p: 4,
      }}
    >
      <Flex
        sx={{
          flexDirection: 'column',
          alignItems: 'center',
          gap: 4,
          w: 'full',
          h: 'full',
        }}
      >
        <IAICanvasToolbar />
        <Flex
          sx={{
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            gap: 4,
            w: 'full',
            h: 'full',
          }}
        >
          <Box sx={{ w: 'full', h: 'full', position: 'relative' }}>
            {doesCanvasNeedScaling ? <IAICanvasResizer /> : <IAICanvas />}
            {active && <IAIDropOverlay isOver={isOver} />}
          </Box>
        </Flex>
      </Flex>
    </Box>
  );
};

export default memo(UnifiedCanvasContent);
