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
  GenericCanvasState,
} from 'features/canvas/canvasSlice';

// component
import IAICanvasMaskLines from './IAICanvasMaskLines';
import IAICanvasMaskBrushPreview from './IAICanvasMaskBrushPreview';
import IAICanvasMaskBrushPreviewOutline from './IAICanvasMaskBrushPreviewOutline';
import { Vector2d } from 'konva/lib/types';
import IAICanvasBoundingBoxPreview from './IAICanvasBoundingBoxPreview';
import { useToast } from '@chakra-ui/react';
import useCanvasHotkeys from './hooks/useCanvasHotkeys';
import _ from 'lodash';
import { createSelector } from '@reduxjs/toolkit';
import { activeTabNameSelector } from 'features/options/optionsSelectors';
import IAICanvasMaskCompositer from './IAICanvasMaskCompositer';
import IAICanvasBoundingBoxPreviewOverlay from './IAICanvasBoundingBoxPreviewOverlay';
import useCanvasWheel from './hooks/useCanvasZoom';
import useCanvasMouseDown from './hooks/useCanvasMouseDown';
import useCanvasMouseUp from './hooks/useCanvasMouseUp';
import useCanvasMouseMove from './hooks/useCanvasMouseMove';
import useCanvasMouseEnter from './hooks/useCanvasMouseEnter';
import useCanvasMouseOut from './hooks/useCanvasMouseOut';
import useCanvasDragMove from './hooks/useCanvasDragMove';
import IAICanvasOutpaintingObjects from './IAICanvasOutpaintingRenderer';

const canvasSelector = createSelector(
  [currentCanvasSelector, baseCanvasImageSelector, activeTabNameSelector],
  (currentCanvas: GenericCanvasState, baseCanvasImage, activeTabName) => {
    const {
      shouldInvertMask,
      shouldShowMask,
      shouldShowCheckboardTransparency,
      stageScale,
      shouldShowBoundingBox,
      shouldShowBoundingBoxFill,
      shouldLockBoundingBox,
      boundingBoxDimensions,
      isTransformingBoundingBox,
      isMouseOverBoundingBox,
      isMovingBoundingBox,
      stageDimensions,
      stageCoordinates,
      isMoveStageKeyHeld,
    } = currentCanvas;

    let stageCursor: string | undefined = '';

    if (isTransformingBoundingBox) {
      stageCursor = undefined;
    } else if (
      isMovingBoundingBox ||
      isMouseOverBoundingBox ||
      isMoveStageKeyHeld
    ) {
      stageCursor = 'move';
    } else if (shouldShowMask) {
      stageCursor = 'none';
    } else {
      stageCursor = 'default';
    }

    return {
      shouldInvertMask,
      shouldShowMask,
      shouldShowCheckboardTransparency,
      stageScale,
      shouldShowBoundingBox,
      shouldShowBoundingBoxFill,
      shouldLockBoundingBox,
      boundingBoxDimensions,
      isTransformingBoundingBox,
      isModifyingBoundingBox: isTransformingBoundingBox || isMovingBoundingBox,
      stageCursor,
      isMouseOverBoundingBox,
      stageDimensions,
      stageCoordinates,
      isMoveStageKeyHeld,
      activeTabName,
      baseCanvasImage,
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
    shouldShowMask,
    shouldShowCheckboardTransparency,
    stageScale,
    shouldShowBoundingBox,
    shouldShowBoundingBoxFill,
    isModifyingBoundingBox,
    stageCursor,
    stageDimensions,
    stageCoordinates,
    isMoveStageKeyHeld,
    boundingBoxDimensions,
    activeTabName,
    baseCanvasImage,
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
  const handleDragMove = useCanvasDragMove();

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
        <div className="canvas-status-text">{`${boundingBoxDimensions.width}x${boundingBoxDimensions.height}`}</div>
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
          onDragMove={handleDragMove}
          onWheel={handleWheel}
          draggable={isMoveStageKeyHeld && activeTabName === 'outpainting'}
        >
          <Layer
            id={'image-layer'}
            ref={canvasImageLayerRef}
            listening={false}
            visible={!shouldInvertMask && !shouldShowCheckboardTransparency}
            imageSmoothingEnabled={false}
          >
            {canvasBgImage && (
              <KonvaImage listening={false} image={canvasBgImage} />
            )}
            <IAICanvasOutpaintingObjects />
          </Layer>
          <Layer id={'mask-layer'} listening={false} visible={shouldShowMask}>
            <IAICanvasMaskLines visible={true} />

            <IAICanvasMaskBrushPreview
              visible={!isModifyingBoundingBox && !isMoveStageKeyHeld}
            />
            <IAICanvasMaskCompositer />

            <IAICanvasMaskBrushPreviewOutline
              visible={!isModifyingBoundingBox && !isMoveStageKeyHeld}
            />

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
          <Layer id={'bounding-box-layer'} visible={shouldShowMask}>
            <IAICanvasBoundingBoxPreviewOverlay
              visible={shouldShowBoundingBoxFill && shouldShowBoundingBox}
            />
            <IAICanvasBoundingBoxPreview visible={shouldShowBoundingBox} />
          </Layer>
        </Stage>
      </div>
    </div>
  );
};

export default IAICanvas;
