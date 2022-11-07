// lib
import {
  MutableRefObject,
  useCallback,
  useEffect,
  useRef,
  useState,
} from 'react';
import Konva from 'konva';
import { Layer, Stage } from 'react-konva';
import { Image as KonvaImage } from 'react-konva';
import { Stage as StageType } from 'konva/lib/Stage';

// app
import { RootState, useAppDispatch, useAppSelector } from 'app/store';
import {
  addLine,
  addPointToCurrentLine,
  CanvasState,
  clearImageToInpaint,
  currentCanvasSelector,
  GenericCanvasState,
  setCursorPosition,
  setIsDrawing,
  setStageCoordinates,
  setStageScale,
} from 'features/canvas/canvasSlice';

// component
import IAICanvasLines from './IAICanvasLines';
import IAICanvasBrushPreview from './IAICanvasBrushPreview';
import IAICanvasBrushPreviewOutline from './IAICanvasBrushPreviewOutline';
import { Vector2d } from 'konva/lib/types';
import getScaledCursorPosition from './util/getScaledCursorPosition';
import IAICanvasBoundingBoxPreview from './IAICanvasBoundingBoxPreview';
import { KonvaEventObject } from 'konva/lib/Node';
import { useToast } from '@chakra-ui/react';
import useCanvasHotkeys from './hooks/useCanvasHotkeys';
import {
  CANVAS_SCALE_BY,
  MAX_CANVAS_SCALE,
  MIN_CANVAS_SCALE,
} from './util/constants';
import _ from 'lodash';
import { createSelector } from '@reduxjs/toolkit';
import { activeTabNameSelector } from 'features/options/optionsSelectors';
import IAICanvasImage from './IAICanvasImage';
import IAICanvasMaskCompositer from './IAICanvasMaskCompositer';
import IAICanvasBoundingBoxPreviewOverlay from './IAICanvasBoundingBoxPreviewOverlay';

const canvasSelector = createSelector(
  [
    currentCanvasSelector,
    activeTabNameSelector,
    (state: RootState) => state.canvas,
  ],
  (currentCanvas: GenericCanvasState, activeTabName, canvas: CanvasState) => {
    const {
      tool,
      brushSize,
      maskColor,
      shouldInvertMask,
      shouldShowMask,
      shouldShowCheckboardTransparency,
      imageToInpaint,
      stageScale,
      shouldShowBoundingBox,
      shouldShowBoundingBoxFill,
      isDrawing,
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
      tool,
      brushSize,
      shouldInvertMask,
      shouldShowMask,
      shouldShowCheckboardTransparency,
      maskColor,
      imageToInpaint,
      stageScale,
      shouldShowBoundingBox,
      shouldShowBoundingBoxFill,
      isDrawing,
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
      outpaintingSession:
        canvas.currentCanvas === 'outpainting'
          ? canvas.outpainting.session
          : undefined,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: (a, b) => {
        const { imageToInpaint: a_imageToInpaint, ...a_rest } = a;
        const { imageToInpaint: b_imageToInpaint, ...b_rest } = b;
        return (
          _.isEqual(a_rest, b_rest) && a_imageToInpaint == b_imageToInpaint
        );
      },
    },
  }
);

// Use a closure allow other components to use these things... not ideal...
export let stageRef: MutableRefObject<StageType | null>;
export let maskLayerRef: MutableRefObject<Konva.Layer | null>;
export let canvasImageLayerRef: MutableRefObject<Konva.Layer | null>;
export let inpaintingImageElementRef: MutableRefObject<HTMLImageElement | null>;

const IAICanvas = () => {
  const dispatch = useAppDispatch();

  const {
    tool,
    brushSize,
    shouldInvertMask,
    shouldShowMask,
    shouldShowCheckboardTransparency,
    maskColor,
    imageToInpaint,
    stageScale,
    shouldShowBoundingBox,
    shouldShowBoundingBoxFill,
    isDrawing,
    isModifyingBoundingBox,
    stageCursor,
    stageDimensions,
    stageCoordinates,
    isMoveStageKeyHeld,
    boundingBoxDimensions,
    activeTabName,
    outpaintingSession,
  } = useAppSelector(canvasSelector);

  useCanvasHotkeys();

  const toast = useToast();
  // useCacher();
  // set the closure'd refs
  stageRef = useRef<StageType>(null);
  maskLayerRef = useRef<Konva.Layer>(null);
  canvasImageLayerRef = useRef<Konva.Layer>(null);
  inpaintingImageElementRef = useRef<HTMLImageElement>(null);

  const lastCursorPosition = useRef<Vector2d>({ x: 0, y: 0 });

  // Use refs for values that do not affect rendering, other values in redux
  const didMouseMoveRef = useRef<boolean>(false);

  // Load the image into this
  const [canvasBgImage, setCanvasBgImage] = useState<HTMLImageElement | null>(
    null
  );

  // Load the image and set the options panel width & height
  useEffect(() => {
    if (imageToInpaint) {
      const image = new Image();
      image.onload = () => {
        inpaintingImageElementRef.current = image;
        setCanvasBgImage(image);
      };
      image.onerror = () => {
        toast({
          title: 'Unable to Load Image',
          description: `Image ${imageToInpaint.url} failed to load`,
          status: 'error',
          isClosable: true,
        });
        dispatch(clearImageToInpaint());
      };
      image.src = imageToInpaint.url;
    } else {
      setCanvasBgImage(null);
    }
  }, [imageToInpaint, dispatch, stageScale, toast]);

  /**
   *
   * Canvas onMouseDown
   *
   */
  const handleMouseDown = useCallback(
    (e: KonvaEventObject<MouseEvent>) => {
      if (!stageRef.current) return;

      const scaledCursorPosition = getScaledCursorPosition(stageRef.current);

      if (
        !scaledCursorPosition ||
        !maskLayerRef.current ||
        isModifyingBoundingBox ||
        isMoveStageKeyHeld
      )
        return;
      e.evt.preventDefault();
      dispatch(setIsDrawing(true));

      // Add a new line starting from the current cursor position.
      dispatch(
        addLine({
          tool,
          strokeWidth: brushSize / 2,
          points: [scaledCursorPosition.x, scaledCursorPosition.y],
        })
      );
    },
    [dispatch, brushSize, tool, isModifyingBoundingBox, isMoveStageKeyHeld]
  );

  /**
   *
   * Canvas onMouseMove
   *
   */
  const handleMouseMove = useCallback(() => {
    if (!stageRef.current) return;

    const scaledCursorPosition = getScaledCursorPosition(stageRef.current);

    if (!scaledCursorPosition) return;

    dispatch(setCursorPosition(scaledCursorPosition));

    if (!maskLayerRef.current) {
      return;
    }

    lastCursorPosition.current = scaledCursorPosition;

    if (!isDrawing || isModifyingBoundingBox || isMoveStageKeyHeld) return;

    didMouseMoveRef.current = true;
    // Extend the current line
    dispatch(
      addPointToCurrentLine([scaledCursorPosition.x, scaledCursorPosition.y])
    );
  }, [dispatch, isDrawing, isModifyingBoundingBox, isMoveStageKeyHeld]);

  /**
   *
   * Canvas onMouseUp
   *
   */
  const handleMouseUp = useCallback(() => {
    if (!didMouseMoveRef.current && isDrawing && stageRef.current) {
      const scaledCursorPosition = getScaledCursorPosition(stageRef.current);

      if (
        !scaledCursorPosition ||
        !maskLayerRef.current ||
        isModifyingBoundingBox ||
        isMoveStageKeyHeld
      )
        return;

      /**
       * Extend the current line.
       * In this case, the mouse didn't move, so we append the same point to
       * the line's existing points. This allows the line to render as a circle
       * centered on that point.
       */
      dispatch(
        addPointToCurrentLine([scaledCursorPosition.x, scaledCursorPosition.y])
      );
    } else {
      didMouseMoveRef.current = false;
    }
    dispatch(setIsDrawing(false));
  }, [dispatch, isDrawing, isModifyingBoundingBox, isMoveStageKeyHeld]);

  /**
   *
   * Canvas onMouseOut
   *
   */
  const handleMouseOutCanvas = useCallback(() => {
    dispatch(setCursorPosition(null));
    dispatch(setIsDrawing(false));
  }, [dispatch]);

  /**
   *
   * Canvas onMouseEnter
   *
   */
  const handleMouseEnter = useCallback(
    (e: KonvaEventObject<MouseEvent>) => {
      if (e.evt.buttons === 1) {
        if (!stageRef.current) return;

        const scaledCursorPosition = getScaledCursorPosition(stageRef.current);

        if (
          !scaledCursorPosition ||
          !maskLayerRef.current ||
          isModifyingBoundingBox ||
          isMoveStageKeyHeld
        )
          return;

        dispatch(setIsDrawing(true));

        // Add a new line starting from the current cursor position.
        dispatch(
          addLine({
            tool,
            strokeWidth: brushSize / 2,
            points: [scaledCursorPosition.x, scaledCursorPosition.y],
          })
        );
      }
    },
    [dispatch, brushSize, tool, isModifyingBoundingBox, isMoveStageKeyHeld]
  );

  const handleWheel = (e: KonvaEventObject<WheelEvent>) => {
    // stop default scrolling
    if (activeTabName !== 'outpainting') return;

    e.evt.preventDefault();

    // const oldScale = stageRef.current.scaleX();
    if (!stageRef.current || isMoveStageKeyHeld) return;

    const cursorPos = stageRef.current.getPointerPosition();

    if (!cursorPos) return;

    const mousePointTo = {
      x: (cursorPos.x - stageRef.current.x()) / stageScale,
      y: (cursorPos.y - stageRef.current.y()) / stageScale,
    };

    let delta = e.evt.deltaY;

    // when we zoom on trackpad, e.evt.ctrlKey is true
    // in that case lets revert direction
    if (e.evt.ctrlKey) {
      delta = -delta;
    }

    const newScale = _.clamp(
      stageScale * CANVAS_SCALE_BY ** delta,
      MIN_CANVAS_SCALE,
      MAX_CANVAS_SCALE
    );

    const newPos = {
      x: cursorPos.x - mousePointTo.x * newScale,
      y: cursorPos.y - mousePointTo.y * newScale,
    };

    dispatch(setStageScale(newScale));
    dispatch(setStageCoordinates(newPos));
  };

  const handleDragStage = (e: KonvaEventObject<MouseEvent>) => {
    if (!isMoveStageKeyHeld) return;
    dispatch(setStageCoordinates(e.target.getPosition()));
  };

  return (
    <div className="inpainting-canvas-container">
      <div className="inpainting-canvas-wrapper">
        <div className="canvas-status-text">{`${boundingBoxDimensions.width}x${boundingBoxDimensions.height}`}</div>
        {canvasBgImage && (
          <Stage
            width={stageDimensions.width}
            height={stageDimensions.height}
            scale={{ x: stageScale, y: stageScale }}
            x={stageCoordinates.x}
            y={stageCoordinates.y}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseEnter={handleMouseEnter}
            onMouseUp={handleMouseUp}
            onMouseOut={handleMouseOutCanvas}
            onMouseLeave={handleMouseOutCanvas}
            onDragMove={handleDragStage}
            draggable={isMoveStageKeyHeld && activeTabName === 'outpainting'}
            onWheel={handleWheel}
            style={{ ...(stageCursor ? { cursor: stageCursor } : {}) }}
            className="inpainting-canvas-stage checkerboard"
            ref={stageRef}
          >
            <Layer
              id={'image-layer'}
              ref={canvasImageLayerRef}
              listening={false}
              visible={!shouldInvertMask && !shouldShowCheckboardTransparency}
            >
              <KonvaImage listening={false} image={canvasBgImage} />
              {outpaintingSession &&
                _.map(outpaintingSession, (region, i) =>
                  region.images.length > 0 ? (
                    <IAICanvasImage
                      key={i}
                      x={region.x}
                      y={region.y}
                      url={region.images[region.selectedImageIndex].url}
                    />
                  ) : null
                )}
            </Layer>
            <Layer
              id={'mask-layer'}
              listening={false}
              ref={maskLayerRef}
              visible={shouldShowMask}
            >
              <IAICanvasLines visible={true} />

              <IAICanvasBrushPreview
                visible={!isModifyingBoundingBox && !isMoveStageKeyHeld}
              />
              <IAICanvasMaskCompositer />

              <IAICanvasBrushPreviewOutline
                visible={!isModifyingBoundingBox && !isMoveStageKeyHeld}
              />

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
                visible={!shouldInvertMask && shouldShowCheckboardTransparency}
              />
            </Layer>
            <Layer id={'bounding-box-layer'} visible={shouldShowMask}>
              <IAICanvasBoundingBoxPreviewOverlay
                visible={shouldShowBoundingBoxFill && shouldShowBoundingBox}
              />
              <IAICanvasBoundingBoxPreview visible={shouldShowBoundingBox} />
            </Layer>
          </Stage>
        )}
      </div>
    </div>
  );
};

export default IAICanvas;
