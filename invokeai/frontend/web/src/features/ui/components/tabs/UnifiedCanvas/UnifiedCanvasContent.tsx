import { Box, Flex } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAIDropOverlay from 'common/components/IAIDropOverlay';
import IAICanvas from 'features/canvas/components/IAICanvas';
import IAICanvasResizer from 'features/canvas/components/IAICanvasResizer';
import IAICanvasToolbar from 'features/canvas/components/IAICanvasToolbar/IAICanvasToolbar';
import { canvasSelector } from 'features/canvas/store/canvasSelectors';
import { requestCanvasRescale } from 'features/canvas/store/thunks/requestCanvasScale';
import { useDroppableTypesafe } from 'features/dnd/hooks/typesafeHooks';
import { CanvasInitialImageDropData } from 'features/dnd/types';
import { isValidDrop } from 'features/dnd/util/isValidDrop';
import { uiSelector } from 'features/ui/store/uiSelectors';
import { memo, useLayoutEffect } from 'react';
import UnifiedCanvasToolSettingsBeta from './UnifiedCanvasBeta/UnifiedCanvasToolSettingsBeta';
import UnifiedCanvasToolbarBeta from './UnifiedCanvasBeta/UnifiedCanvasToolbarBeta';

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

const droppableData: CanvasInitialImageDropData = {
  id: 'canvas-intial-image',
  actionType: 'SET_CANVAS_INITIAL_IMAGE',
};

const UnifiedCanvasContent = () => {
  const dispatch = useAppDispatch();

  const { doesCanvasNeedScaling, shouldUseCanvasBetaLayout } =
    useAppSelector(selector);

  const {
    isOver,
    setNodeRef: setDroppableRef,
    active,
  } = useDroppableTypesafe({
    id: 'unifiedCanvas',
    data: droppableData,
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
        layerStyle="first"
        ref={setDroppableRef}
        tabIndex={0}
        sx={{
          w: 'full',
          h: 'full',
          p: 4,
          borderRadius: 'base',
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
              {isValidDrop(droppableData, active) && (
                <IAIDropOverlay
                  isOver={isOver}
                  label="Set Canvas Initial Image"
                />
              )}
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
        layerStyle: 'first',
        w: 'full',
        h: 'full',
        p: 4,
        borderRadius: 'base',
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
            {isValidDrop(droppableData, active) && (
              <IAIDropOverlay
                isOver={isOver}
                label="Set Canvas Initial Image"
              />
            )}
          </Box>
        </Flex>
      </Flex>
    </Box>
  );
};

export default memo(UnifiedCanvasContent);
