// lib
import { MutableRefObject, useRef } from 'react';
import Konva from 'konva';
import { Layer, Stage } from 'react-konva';
import { Stage as StageType } from 'konva/lib/Stage';

// app
import { useAppSelector } from 'app/store';
import {
  baseCanvasImageSelector,
  currentCanvasSelector,
  outpaintingCanvasSelector,
} from 'features/canvas/canvasSlice';

// component
import IAICanvasMaskLines from './IAICanvasMaskLines';
import IAICanvasBrushPreview from './IAICanvasBrushPreview';
import { Vector2d } from 'konva/lib/types';
import IAICanvasBoundingBox from './IAICanvasBoundingBox';
import useCanvasHotkeys from './hooks/useCanvasHotkeys';
import _ from 'lodash';
import { createSelector } from '@reduxjs/toolkit';
import { activeTabNameSelector } from 'features/options/optionsSelectors';
import IAICanvasMaskCompositer from './IAICanvasMaskCompositer';
import useCanvasWheel from './hooks/useCanvasZoom';
import useCanvasMouseDown from './hooks/useCanvasMouseDown';
import useCanvasMouseUp from './hooks/useCanvasMouseUp';
import useCanvasMouseMove from './hooks/useCanvasMouseMove';
import useCanvasMouseEnter from './hooks/useCanvasMouseEnter';
import useCanvasMouseOut from './hooks/useCanvasMouseOut';
import useCanvasDragMove from './hooks/useCanvasDragMove';
import IAICanvasObjectRenderer from './IAICanvasObjectRenderer';
import IAICanvasGrid from './IAICanvasGrid';
import IAICanvasIntermediateImage from './IAICanvasIntermediateImage';
import IAICanvasStatusText from './IAICanvasStatusText';

const canvasSelector = createSelector(
  [
    currentCanvasSelector,
    outpaintingCanvasSelector,
    baseCanvasImageSelector,
    activeTabNameSelector,
  ],
  (currentCanvas, outpaintingCanvas, baseCanvasImage, activeTabName) => {
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
    } = currentCanvas;

    const { shouldShowGrid } = outpaintingCanvas;

    let stageCursor: string | undefined = '';

    if (tool === 'move') {
      if (isTransformingBoundingBox) {
        stageCursor = undefined;
      } else if (isMouseOverBoundingBox) {
        stageCursor = 'move';
      } else if (activeTabName === 'outpainting') {
        if (isMovingStage) {
          stageCursor = 'grabbing';
        } else {
          stageCursor = 'grab';
        }
      }
    } else {
      stageCursor = 'none';
    }

    return {
      activeTabName,
      isMaskEnabled,
      isModifyingBoundingBox: isTransformingBoundingBox || isMovingBoundingBox,
      shouldShowBoundingBox,
      shouldShowGrid,
      stageCoordinates,
      stageCursor,
      stageDimensions,
      stageScale,
      tool,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: _.isEqual,
    },
  }
);

// Use a closure allow other components to use these things... not ideal...
export let stageRef: MutableRefObject<StageType | null>;
export let canvasImageLayerRef: MutableRefObject<Konva.Layer | null>;

const IAICanvas = () => {
  const {
    activeTabName,
    isMaskEnabled,
    isModifyingBoundingBox,
    shouldShowBoundingBox,
    shouldShowGrid,
    stageCoordinates,
    stageCursor,
    stageDimensions,
    stageScale,
    tool,
  } = useAppSelector(canvasSelector);

  useCanvasHotkeys();

  // set the closure'd refs
  stageRef = useRef<StageType>(null);
  canvasImageLayerRef = useRef<Konva.Layer>(null);

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
  const handleMouseEnter = useCanvasMouseEnter(stageRef);
  const handleMouseOut = useCanvasMouseOut();
  const { handleDragStart, handleDragMove, handleDragEnd } =
    useCanvasDragMove();

  return (
    <div className="inpainting-canvas-container">
      <div className="inpainting-canvas-wrapper">
        <Stage
          ref={stageRef}
          style={{ ...(stageCursor ? { cursor: stageCursor } : {}) }}
          className="inpainting-canvas-stage checkerboard"
          x={stageCoordinates.x}
          y={stageCoordinates.y}
          width={stageDimensions.width}
          height={stageDimensions.height}
          scale={{ x: stageScale, y: stageScale }}
          onMouseDown={handleMouseDown}
          onMouseEnter={handleMouseEnter}
          onMouseLeave={handleMouseOut}
          onMouseMove={handleMouseMove}
          onMouseOut={handleMouseOut}
          onMouseUp={handleMouseUp}
          onDragStart={handleDragStart}
          onDragMove={handleDragMove}
          onDragEnd={handleDragEnd}
          onWheel={handleWheel}
          listening={tool === 'move' && !isModifyingBoundingBox}
          draggable={
            tool === 'move' &&
            !isModifyingBoundingBox &&
            activeTabName === 'outpainting'
          }
        >
          <Layer id={'grid'} visible={shouldShowGrid}>
            <IAICanvasGrid />
          </Layer>

          <Layer
            id={'image'}
            ref={canvasImageLayerRef}
            listening={false}
            imageSmoothingEnabled={false}
          >
            <IAICanvasObjectRenderer />
            <IAICanvasIntermediateImage />
          </Layer>
          <Layer id={'mask'} visible={isMaskEnabled} listening={false}>
            <IAICanvasMaskLines visible={true} listening={false} />
            <IAICanvasMaskCompositer listening={false} />

            {/* {canvasBgImage && (
              <>
                <KonvaImage
                  image={canvasBgImage}
                  listening={false}
                  globalCompositeOperation="source-in"
                  visible={shouldInvertMask}
                />

                <KonvaImage
                  image={canvasBgImage}
                  listening={false}
                  globalCompositeOperation="source-out"
                  visible={
                    !shouldInvertMask && shouldShowCheckboardTransparency
                  }
                />
              </>
            )} */}
          </Layer>
          <Layer id={'tool'}>
            <IAICanvasBoundingBox visible={shouldShowBoundingBox} />
            <IAICanvasBrushPreview
              visible={tool !== 'move'}
              listening={false}
            />
          </Layer>
        </Stage>
        <IAICanvasStatusText />
      </div>
    </div>
  );
};

export default IAICanvas;
