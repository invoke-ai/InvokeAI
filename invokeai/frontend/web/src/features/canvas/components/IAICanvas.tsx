import { Box, chakra, Flex } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import useCanvasDragMove from 'features/canvas/hooks/useCanvasDragMove';
import useCanvasHotkeys from 'features/canvas/hooks/useCanvasHotkeys';
import useCanvasMouseDown from 'features/canvas/hooks/useCanvasMouseDown';
import useCanvasMouseMove from 'features/canvas/hooks/useCanvasMouseMove';
import useCanvasMouseOut from 'features/canvas/hooks/useCanvasMouseOut';
import useCanvasMouseUp from 'features/canvas/hooks/useCanvasMouseUp';
import useCanvasWheel from 'features/canvas/hooks/useCanvasZoom';
import {
  $canvasBaseLayer,
  $canvasStage,
  $isModifyingBoundingBox,
  $isMouseOverBoundingBox,
  $isMovingStage,
  $isTransformingBoundingBox,
  $tool,
} from 'features/canvas/store/canvasNanostore';
import { isStagingSelector } from 'features/canvas/store/canvasSelectors';
import { canvasResized, selectCanvasSlice } from 'features/canvas/store/canvasSlice';
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

const selector = createMemoizedSelector(selectCanvasSlice, (canvas) => {
  return {
    stageCoordinates: canvas.stageCoordinates,
    stageDimensions: canvas.stageDimensions,
  };
});

const ChakraStage = chakra(Stage, {
  shouldForwardProp: (prop) => !['sx'].includes(prop),
});

const IAICanvas = () => {
  const isStaging = useAppSelector(isStagingSelector);
  const isMaskEnabled = useAppSelector((s) => s.canvas.isMaskEnabled);
  const shouldShowBoundingBox = useAppSelector((s) => s.canvas.shouldShowBoundingBox);
  const shouldShowGrid = useAppSelector((s) => s.canvas.shouldShowGrid);
  const stageScale = useAppSelector((s) => s.canvas.stageScale);
  const shouldShowIntermediates = useAppSelector((s) => s.canvas.shouldShowIntermediates);
  const shouldAntialias = useAppSelector((s) => s.canvas.shouldAntialias);
  const shouldRestrictStrokesToBox = useAppSelector((s) => s.canvas.shouldRestrictStrokesToBox);
  const { stageCoordinates, stageDimensions } = useAppSelector(selector);
  const dispatch = useAppDispatch();
  const containerRef = useRef<HTMLDivElement>(null);
  const stageRef = useRef<Konva.Stage | null>(null);
  const canvasBaseLayerRef = useRef<Konva.Layer | null>(null);
  const isModifyingBoundingBox = useStore($isModifyingBoundingBox);
  const isMovingStage = useStore($isMovingStage);
  const isTransformingBoundingBox = useStore($isTransformingBoundingBox);
  const isMouseOverBoundingBox = useStore($isMouseOverBoundingBox);
  const tool = useStore($tool);
  useCanvasHotkeys();
  const canvasStageRefCallback = useCallback((stageElement: Konva.Stage) => {
    $canvasStage.set(stageElement);
    stageRef.current = stageElement;
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
  }, [isMouseOverBoundingBox, isMovingStage, isStaging, isTransformingBoundingBox, shouldRestrictStrokesToBox, tool]);

  const canvasBaseLayerRefCallback = useCallback((layerElement: Konva.Layer) => {
    $canvasBaseLayer.set(layerElement);
    canvasBaseLayerRef.current = layerElement;
  }, []);

  const lastCursorPositionRef = useRef<Vector2d>({ x: 0, y: 0 });

  // Use refs for values that do not affect rendering, other values in redux
  const didMouseMoveRef = useRef<boolean>(false);

  const handleWheel = useCanvasWheel(stageRef);
  const handleMouseDown = useCanvasMouseDown(stageRef);
  const handleMouseUp = useCanvasMouseUp(stageRef, didMouseMoveRef);
  const handleMouseMove = useCanvasMouseMove(stageRef, didMouseMoveRef, lastCursorPositionRef);
  const { handleDragStart, handleDragMove, handleDragEnd } = useCanvasDragMove();
  const handleMouseOut = useCanvasMouseOut();
  const handleContextMenu = useCallback((e: KonvaEventObject<MouseEvent>) => e.evt.preventDefault(), []);

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
    <Flex id="canvas-container" ref={containerRef} position="relative" height="100%" width="100%" borderRadius="base">
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

          <Layer id="base" ref={canvasBaseLayerRefCallback} listening={false} imageSmoothingEnabled={shouldAntialias}>
            <IAICanvasObjectRenderer />
          </Layer>
          <Layer id="mask" visible={isMaskEnabled && !isStaging} listening={false}>
            <IAICanvasMaskLines visible={true} listening={false} />
            <IAICanvasMaskCompositer listening={false} />
          </Layer>
          <Layer listening={false}>
            <IAICanvasBoundingBoxOverlay />
          </Layer>
          <Layer id="preview" imageSmoothingEnabled={shouldAntialias}>
            {!isStaging && <IAICanvasToolPreview visible={tool !== 'move'} listening={false} />}
            <IAICanvasStagingArea listening={false} visible={isStaging} />
            {shouldShowIntermediates && <IAICanvasIntermediateImage />}
            <IAICanvasBoundingBox visible={shouldShowBoundingBox && !isStaging} />
          </Layer>
        </ChakraStage>
      </Box>
      <IAICanvasStatusText />
      <IAICanvasStagingAreaToolbar />
    </Flex>
  );
};

export default memo(IAICanvas);
