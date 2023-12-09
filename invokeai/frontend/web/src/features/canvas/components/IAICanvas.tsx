import { Box, chakra, Flex } from '@chakra-ui/react';
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
import { isStagingSelector } from 'features/canvas/store/canvasSelectors';
import { canvasResized } from 'features/canvas/store/canvasSlice';
import {
  setCanvasBaseLayer,
  setCanvasStage,
} from 'features/canvas/util/konvaInstanceProvider';
import Konva from 'konva';
import { KonvaEventObject } from 'konva/lib/Node';
import { Vector2d } from 'konva/lib/types';
import { memo, useCallback, useEffect, useRef } from 'react';
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
  }
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
  const dispatch = useAppDispatch();
  const containerRef = useRef<HTMLDivElement>(null);
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

  const handleContextMenu = useCallback(
    (e: KonvaEventObject<MouseEvent>) => e.evt.preventDefault(),
    []
  );

  useEffect(() => {
    if (!containerRef.current) {
      return;
    }
    const resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        if (entry.contentBoxSize) {
          const { width, height } = entry.contentRect;
          dispatch(canvasResized({ width, height }));
        }
      }
    });

    resizeObserver.observe(containerRef.current);
    const { width, height } = containerRef.current.getBoundingClientRect();
    dispatch(canvasResized({ width, height }));

    return () => {
      resizeObserver.disconnect();
    };
  }, [dispatch]);

  return (
    <Flex
      id="canvas-container"
      ref={containerRef}
      sx={{
        position: 'relative',
        height: '100%',
        width: '100%',
        borderRadius: 'base',
      }}
    >
      <Box
        sx={{
          position: 'absolute',
          // top: 0,
          // insetInlineStart: 0,
        }}
      >
        <ChakraStage
          tabIndex={-1}
          ref={canvasStageRefCallback}
          sx={{
            outline: 'none',
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
          onContextMenu={handleContextMenu}
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
          <Layer
            id="mask"
            visible={isMaskEnabled && !isStaging}
            listening={false}
          >
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
      </Box>
      <IAICanvasStatusText />
      <IAICanvasStagingAreaToolbar />
    </Flex>
  );
};

export default memo(IAICanvas);
