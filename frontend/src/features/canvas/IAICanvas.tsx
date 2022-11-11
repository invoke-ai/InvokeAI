// lib
import { MutableRefObject, useEffect, useRef, useState } from 'react';
import Konva from 'konva';
import { Layer, Stage } from 'react-konva';
import { Image as KonvaImage } from 'react-konva';
import { Stage as StageType } from 'konva/lib/Stage';

// app
import { useAppDispatch, useAppSelector } from 'app/store';
import {
  baseCanvasImageSelector,
  clearImageToInpaint,
  currentCanvasSelector,
  outpaintingCanvasSelector,
} from 'features/canvas/canvasSlice';

// component
import IAICanvasMaskLines from './IAICanvasMaskLines';
import IAICanvasBrushPreview from './IAICanvasBrushPreview';
import { Vector2d } from 'konva/lib/types';
import IAICanvasBoundingBoxPreview from './IAICanvasBoundingBoxPreview';
import { useToast } from '@chakra-ui/react';
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
import IAICanvasOutpaintingObjects from './IAICanvasOutpaintingObjects';
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
      shouldInvertMask,
      isMaskEnabled,
      shouldShowCheckboardTransparency,
      stageScale,
      shouldShowBoundingBox,
      shouldLockBoundingBox,
      isTransformingBoundingBox,
      isMouseOverBoundingBox,
      isMovingBoundingBox,
      stageDimensions,
      stageCoordinates,
      isMoveStageKeyHeld,
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
      } else {
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
      shouldInvertMask,
      isMaskEnabled,
      shouldShowCheckboardTransparency,
      stageScale,
      shouldShowBoundingBox,
      shouldLockBoundingBox,
      shouldShowGrid,
      isTransformingBoundingBox,
      isModifyingBoundingBox: isTransformingBoundingBox || isMovingBoundingBox,
      stageCursor,
      isMouseOverBoundingBox,
      stageDimensions,
      stageCoordinates,
      isMoveStageKeyHeld,
      activeTabName,
      baseCanvasImage,
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
export let inpaintingImageElementRef: MutableRefObject<HTMLImageElement | null>;

const IAICanvas = () => {
  const dispatch = useAppDispatch();

  const {
    shouldInvertMask,
    isMaskEnabled,
    shouldShowCheckboardTransparency,
    stageScale,
    shouldShowBoundingBox,
    isModifyingBoundingBox,
    stageCursor,
    stageDimensions,
    stageCoordinates,
    shouldShowGrid,
    activeTabName,
    baseCanvasImage,
    tool,
  } = useAppSelector(canvasSelector);

  useCanvasHotkeys();

  const toast = useToast();
  // set the closure'd refs
  stageRef = useRef<StageType>(null);
  canvasImageLayerRef = useRef<Konva.Layer>(null);
  inpaintingImageElementRef = useRef<HTMLImageElement>(null);

  const lastCursorPositionRef = useRef<Vector2d>({ x: 0, y: 0 });

  // Use refs for values that do not affect rendering, other values in redux
  const didMouseMoveRef = useRef<boolean>(false);

  // Load the image into this
  const [canvasBgImage, setCanvasBgImage] = useState<HTMLImageElement | null>(
    null
  );

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

  // Load the image and set the options panel width & height
  useEffect(() => {
    if (baseCanvasImage) {
      const image = new Image();
      image.onload = () => {
        inpaintingImageElementRef.current = image;
        setCanvasBgImage(image);
      };
      image.onerror = () => {
        toast({
          title: 'Unable to Load Image',
          description: `Image ${baseCanvasImage.url} failed to load`,
          status: 'error',
          isClosable: true,
        });
        dispatch(clearImageToInpaint());
      };
      image.src = baseCanvasImage.url;
    } else {
      setCanvasBgImage(null);
    }
  }, [baseCanvasImage, dispatch, stageScale, toast]);

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
          listening={
            tool === 'move' &&
            !isModifyingBoundingBox &&
            activeTabName === 'outpainting'
          }
          draggable={
            tool === 'move' &&
            !isModifyingBoundingBox &&
            activeTabName === 'outpainting'
          }
        >
          <Layer visible={shouldShowGrid}>
            <IAICanvasGrid />
          </Layer>

          <Layer
            id={'image-layer'}
            ref={canvasImageLayerRef}
            listening={false}
            imageSmoothingEnabled={false}
          >
            <IAICanvasOutpaintingObjects />
            <IAICanvasIntermediateImage />
          </Layer>
          <Layer id={'mask-layer'} visible={isMaskEnabled} listening={false}>
            <IAICanvasMaskLines visible={true} listening={false} />

            <IAICanvasMaskCompositer listening={false} />

            {canvasBgImage && (
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
            )}
          </Layer>
          <Layer id={'preview-layer'}>
            <IAICanvasBoundingBoxPreview visible={shouldShowBoundingBox} />
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
