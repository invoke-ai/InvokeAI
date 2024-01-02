import { Box, chakra, Flex } from '@chakra-ui/react';
import { useStore } from '@nanostores/react';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import useCanvasDragMove from 'features/canvas/hooks/useCanvasDragMove';
import useCanvasHotkeys from 'features/canvas/hooks/useCanvasHotkeys';
import useCanvasMouseDown from 'features/canvas/hooks/useCanvasMouseDown';
import useCanvasMouseMove from 'features/canvas/hooks/useCanvasMouseMove';
import useCanvasMouseOut from 'features/canvas/hooks/useCanvasMouseOut';
import useCanvasMouseUp from 'features/canvas/hooks/useCanvasMouseUp';
import useCanvasWheel from 'features/canvas/hooks/useCanvasZoom';
import {
  $isModifyingBoundingBox,
  $isMouseOverBoundingBox,
  $isMovingStage,
  $isTransformingBoundingBox,
} from 'features/canvas/store/canvasNanostore';
import { isStagingSelector } from 'features/canvas/store/canvasSelectors';
import { canvasResized } from 'features/canvas/store/canvasSlice';
import {
  setCanvasBaseLayer,
  setCanvasStage,
} from 'features/canvas/util/konvaInstanceProvider';
import type Konva from 'konva';
import type { KonvaEventObject } from 'konva/lib/Node';
import type { Vector2d } from 'konva/lib/types';
import { memo, useCallback, useEffect, useMemo, useRef } from 'react';
import { Layer, Stage } from 'react-konva';

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

const selector = createMemoizedSelector(
  [stateSelector, isStagingSelector],
  ({ canvas }, isStaging) => {
    const {
      isMaskEnabled,
      stageScale,
      shouldShowBoundingBox,
      stageDimensions,
      stageCoordinates,
      tool,
      shouldShowIntermediates,
      shouldRestrictStrokesToBox,
      shouldShowGrid,
      shouldAntialias,
    } = canvas;

    return {
      isMaskEnabled,
      shouldShowBoundingBox,
      shouldShowGrid,
      stageCoordinates,
      stageDimensions,
      stageScale,
      tool,
      isStaging,
      shouldShowIntermediates,
      shouldAntialias,
      shouldRestrictStrokesToBox,
    };
  }
);

const ChakraStage = chakra(Stage, {
  shouldForwardProp: (prop) => !['sx'].includes(prop),
});

const IAICanvas = () => {
  const {
    isMaskEnabled,
    shouldShowBoundingBox,
    shouldShowGrid,
    stageCoordinates,
    stageDimensions,
    stageScale,
    tool,
    isStaging,
    shouldShowIntermediates,
    shouldAntialias,
    shouldRestrictStrokesToBox,
  } = useAppSelector(selector);
  useCanvasHotkeys();
  const dispatch = useAppDispatch();
  const containerRef = useRef<HTMLDivElement>(null);
  const stageRef = useRef<Konva.Stage | null>(null);
  const canvasBaseLayerRef = useRef<Konva.Layer | null>(null);
  const isModifyingBoundingBox = useStore($isModifyingBoundingBox);
  const isMovingStage = useStore($isMovingStage);
  const isTransformingBoundingBox = useStore($isTransformingBoundingBox);
  const isMouseOverBoundingBox = useStore($isMouseOverBoundingBox);
  const canvasStageRefCallback = useCallback((el: Konva.Stage) => {
    setCanvasStage(el as Konva.Stage);
    stageRef.current = el;
  }, []);
  const stageCursor = useMemo(() => {
    if (tool === 'move' || isStaging) {
      if (isMovingStage) {
        return 'grabbing';
      } else {
        return 'grab';
      }
    } else if (isTransformingBoundingBox) {
      return undefined;
    } else if (shouldRestrictStrokesToBox && !isMouseOverBoundingBox) {
      return 'default';
    }
    return 'none';
  }, [
    isMouseOverBoundingBox,
    isMovingStage,
    isStaging,
    isTransformingBoundingBox,
    shouldRestrictStrokesToBox,
    tool,
  ]);
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
  const { handleDragStart, handleDragMove, handleDragEnd } =
    useCanvasDragMove();
  const handleMouseOut = useCanvasMouseOut();
  const handleContextMenu = useCallback(
    (e: KonvaEventObject<MouseEvent>) => e.evt.preventDefault(),
    []
  );

  useEffect(() => {
    if (!containerRef.current) {
      return;
    }
    const resizeObserver = new ResizeObserver(() => {
      if (!containerRef.current) {
        return;
      }
      const { width, height } = containerRef.current.getBoundingClientRect();
      dispatch(canvasResized({ width, height }));
    });

    resizeObserver.observe(containerRef.current);
    const { width, height } = containerRef.current.getBoundingClientRect();
    dispatch(canvasResized({ width, height }));

    return () => {
      resizeObserver.disconnect();
    };
  }, [dispatch]);

  const stageStyles = useMemo(
    () => ({
      outline: 'none',
      overflow: 'hidden',
      cursor: stageCursor ? stageCursor : undefined,
      canvas: {
        outline: 'none',
      },
    }),
    [stageCursor]
  );

  const scale = useMemo(() => ({ x: stageScale, y: stageScale }), [stageScale]);

  return (
    <Flex
      id="canvas-container"
      ref={containerRef}
      position="relative"
      height="100%"
      width="100%"
      borderRadius="base"
    >
      <Box position="absolute">
        <ChakraStage
          tabIndex={-1}
          ref={canvasStageRefCallback}
          sx={stageStyles}
          x={stageCoordinates.x}
          y={stageCoordinates.y}
          width={stageDimensions.width}
          height={stageDimensions.height}
          scale={scale}
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
          onContextMenu={handleContextMenu}
          onWheel={handleWheel}
          draggable={(tool === 'move' || isStaging) && !isModifyingBoundingBox}
        >
          <Layer id="grid" visible={shouldShowGrid} listening={false}>
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
          <Layer
            id="mask"
            visible={isMaskEnabled && !isStaging}
            listening={false}
          >
            <IAICanvasMaskLines visible={true} listening={false} />
            <IAICanvasMaskCompositer listening={false} />
          </Layer>
          <Layer listening={false}>
            <IAICanvasBoundingBoxOverlay />
          </Layer>
          <Layer id="preview" imageSmoothingEnabled={shouldAntialias}>
            {!isStaging && (
              <IAICanvasToolPreview
                visible={tool !== 'move'}
                listening={false}
              />
            )}
            <IAICanvasStagingArea listening={false} visible={isStaging} />
            {shouldShowIntermediates && <IAICanvasIntermediateImage />}
            <IAICanvasBoundingBox
              visible={shouldShowBoundingBox && !isStaging}
            />
          </Layer>
        </ChakraStage>
      </Box>
      <IAICanvasStatusText />
      <IAICanvasStagingAreaToolbar />
    </Flex>
  );
};

export default memo(IAICanvas);
