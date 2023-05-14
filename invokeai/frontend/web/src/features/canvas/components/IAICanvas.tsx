import { Box, chakra, Flex } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import {
  canvasSelector,
  isStagingSelector,
} from 'features/canvas/store/canvasSelectors';
import Konva from 'konva';
import { KonvaEventObject } from 'konva/lib/Node';
import { Vector2d } from 'konva/lib/types';
import { isEqual } from 'lodash-es';

import { useCallback, useRef } from 'react';
import { Layer, Stage } from 'react-konva';
import useCanvasDragMove from '../hooks/useCanvasDragMove';
import useCanvasHotkeys from '../hooks/useCanvasHotkeys';
import useCanvasMouseDown from '../hooks/useCanvasMouseDown';
import useCanvasMouseMove from '../hooks/useCanvasMouseMove';
import useCanvasMouseOut from '../hooks/useCanvasMouseOut';
import useCanvasMouseUp from '../hooks/useCanvasMouseUp';
import useCanvasWheel from '../hooks/useCanvasZoom';
import {
  setCanvasBaseLayer,
  setCanvasStage,
} from '../util/konvaInstanceProvider';
import IAICanvasBoundingBoxOverlay from './IAICanvasBoundingBoxOverlay';
import IAICanvasGrid from './IAICanvasGrid';
import IAICanvasIntermediateImage from './IAICanvasIntermediateImage';
import IAICanvasMaskCompositer from './IAICanvasMaskCompositer';
import IAICanvasMaskLines from './IAICanvasMaskLines';
import IAICanvasObjectRenderer from './IAICanvasObjectRenderer';
import IAICanvasStagingArea from './IAICanvasStagingArea';
import IAICanvasStagingAreaToolbar from './IAICanvasStagingAreaToolbar';
import IAICanvasStatusText from './IAICanvasStatusText';
import IAICanvasBoundingBox from './IAICanvasToolbar/IAICanvasBoundingBox';
import IAICanvasToolPreview from './IAICanvasToolPreview';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';

const selector = createSelector(
  [canvasSelector, isStagingSelector],
  (canvas, isStaging) => {
    const {
      isMaskEnabled,
      stageScale,
      shouldShowBoundingBox,
      isTransformingBoundingBox,
      isMouseOverBoundingBox,
      isMovingBoundingBox,
      stageDimensions,
      stageCoordinates,
      tool,
      isMovingStage,
      shouldShowIntermediates,
      shouldShowGrid,
      shouldRestrictStrokesToBox,
      shouldAntialias,
    } = canvas;

    let stageCursor: string | undefined = 'none';

    if (tool === 'move' || isStaging) {
      if (isMovingStage) {
        stageCursor = 'grabbing';
      } else {
        stageCursor = 'grab';
      }
    } else if (isTransformingBoundingBox) {
      stageCursor = undefined;
    } else if (shouldRestrictStrokesToBox && !isMouseOverBoundingBox) {
      stageCursor = 'default';
    }

    return {
      isMaskEnabled,
      isModifyingBoundingBox: isTransformingBoundingBox || isMovingBoundingBox,
      shouldShowBoundingBox,
      shouldShowGrid,
      stageCoordinates,
      stageCursor,
      stageDimensions,
      stageScale,
      tool,
      isStaging,
      shouldShowIntermediates,
      shouldAntialias,
    };
  },
  defaultSelectorOptions
);

const ChakraStage = chakra(Stage, {
  shouldForwardProp: (prop) => !['sx'].includes(prop),
});

const IAICanvas = () => {
  const {
    isMaskEnabled,
    isModifyingBoundingBox,
    shouldShowBoundingBox,
    shouldShowGrid,
    stageCoordinates,
    stageCursor,
    stageDimensions,
    stageScale,
    tool,
    isStaging,
    shouldShowIntermediates,
    shouldAntialias,
  } = useAppSelector(selector);
  useCanvasHotkeys();

  const stageRef = useRef<Konva.Stage | null>(null);
  const canvasBaseLayerRef = useRef<Konva.Layer | null>(null);

  const canvasStageRefCallback = useCallback((el: Konva.Stage) => {
    setCanvasStage(el as Konva.Stage);
    stageRef.current = el;
  }, []);

  const canvasBaseLayerRefCallback = useCallback((el: Konva.Layer) => {
    setCanvasBaseLayer(el as Konva.Layer);
    canvasBaseLayerRef.current = el;
  }, []);

  const lastCursorPositionRef = useRef<Vector2d>({ x: 0, y: 0 });

  // Use refs for values that do not affect rendering, other values in redux
  const didMouseMoveRef = useRef<boolean>(false);

  const handleWheel = useCanvasWheel(stageRef);
  const handleMouseDown = useCanvasMouseDown(stageRef);
  const handleMouseUp = useCanvasMouseUp(stageRef, didMouseMoveRef);
  const handleMouseMove = useCanvasMouseMove(
    stageRef,
    didMouseMoveRef,
    lastCursorPositionRef
  );
  const handleMouseOut = useCanvasMouseOut();
  const { handleDragStart, handleDragMove, handleDragEnd } =
    useCanvasDragMove();

  return (
    <Flex
      sx={{
        position: 'relative',
        height: '100%',
        width: '100%',
        borderRadius: 'base',
      }}
    >
      <Box sx={{ position: 'relative' }}>
        <ChakraStage
          tabIndex={-1}
          ref={canvasStageRefCallback}
          sx={{
            outline: 'none',
            // boxShadow: '0px 0px 0px 1px var(--border-color-light)',
            overflow: 'hidden',
            cursor: stageCursor ? stageCursor : undefined,
            canvas: {
              outline: 'none',
            },
          }}
          x={stageCoordinates.x}
          y={stageCoordinates.y}
          width={stageDimensions.width}
          height={stageDimensions.height}
          scale={{ x: stageScale, y: stageScale }}
          onTouchStart={handleMouseDown}
          onTouchMove={handleMouseMove}
          onTouchEnd={handleMouseUp}
          onMouseDown={handleMouseDown}
          onMouseLeave={handleMouseOut}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onDragStart={handleDragStart}
          onDragMove={handleDragMove}
          onDragEnd={handleDragEnd}
          onContextMenu={(e: KonvaEventObject<MouseEvent>) =>
            e.evt.preventDefault()
          }
          onWheel={handleWheel}
          draggable={(tool === 'move' || isStaging) && !isModifyingBoundingBox}
        >
          <Layer id="grid" visible={shouldShowGrid}>
            <IAICanvasGrid />
          </Layer>

          <Layer
            id="base"
            ref={canvasBaseLayerRefCallback}
            listening={false}
            imageSmoothingEnabled={shouldAntialias}
          >
            <IAICanvasObjectRenderer />
          </Layer>
          <Layer id="mask" visible={isMaskEnabled} listening={false}>
            <IAICanvasMaskLines visible={true} listening={false} />
            <IAICanvasMaskCompositer listening={false} />
          </Layer>
          <Layer>
            <IAICanvasBoundingBoxOverlay />
          </Layer>
          <Layer id="preview" imageSmoothingEnabled={shouldAntialias}>
            {!isStaging && (
              <IAICanvasToolPreview
                visible={tool !== 'move'}
                listening={false}
              />
            )}
            <IAICanvasStagingArea visible={isStaging} />
            {shouldShowIntermediates && <IAICanvasIntermediateImage />}
            <IAICanvasBoundingBox
              visible={shouldShowBoundingBox && !isStaging}
            />
          </Layer>
        </ChakraStage>
        <IAICanvasStatusText />
        <IAICanvasStagingAreaToolbar />
      </Box>
    </Flex>
  );
};

export default IAICanvas;
