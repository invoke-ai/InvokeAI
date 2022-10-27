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
import { useAppDispatch, useAppSelector } from '../../../app/store';
import {
  addLine,
  addPointToCurrentLine,
  setBoundingBoxCoordinate,
  setCursorPosition,
  setIsDrawing,
  setIsMovingBoundingBox,
} from './inpaintingSlice';
import { inpaintingCanvasSelector } from './inpaintingSliceSelectors';

// component
import InpaintingCanvasLines from './components/InpaintingCanvasLines';
import InpaintingCanvasBrushPreview from './components/InpaintingCanvasBrushPreview';
import InpaintingCanvasBrushPreviewOutline from './components/InpaintingCanvasBrushPreviewOutline';
import Cacher from './components/Cacher';
import { Vector2d } from 'konva/lib/types';
import getScaledCursorPosition from './util/getScaledCursorPosition';
import _ from 'lodash';
import InpaintingBoundingBoxPreview, {
  InpaintingBoundingBoxPreviewOverlay,
} from './components/InpaintingBoundingBoxPreview';
import { KonvaEventObject } from 'konva/lib/Node';
import KeyboardEventManager from './components/KeyboardEventManager';

// Use a closure allow other components to use these things... not ideal...
export let stageRef: MutableRefObject<StageType | null>;
export let maskLayerRef: MutableRefObject<Konva.Layer | null>;
export let inpaintingImageElementRef: MutableRefObject<HTMLImageElement | null>;

const InpaintingCanvas = () => {
  const dispatch = useAppDispatch();

  const {
    tool,
    brushSize,
    shouldInvertMask,
    shouldShowMask,
    shouldShowCheckboardTransparency,
    maskColor,
    imageToInpaint,
    isMovingBoundingBox,
    boundingBoxDimensions,
    canvasDimensions,
    boundingBoxCoordinate,
    stageScale,
    shouldShowBoundingBoxFill,
    isDrawing,
    isBoundingBoxTransforming,
  } = useAppSelector(inpaintingCanvasSelector);

  // set the closure'd refs
  stageRef = useRef<StageType>(null);
  maskLayerRef = useRef<Konva.Layer>(null);
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
      image.src = imageToInpaint.url;
    }
  }, [imageToInpaint, dispatch, stageScale]);

  const handleMouseDown = useCallback(() => {
    if (!stageRef.current) return;

    const scaledCursorPosition = getScaledCursorPosition(stageRef.current);

    if (!scaledCursorPosition || !maskLayerRef.current) return;

    dispatch(setIsDrawing(true));

    // Add a new line starting from the current cursor position.
    dispatch(
      addLine({
        tool,
        strokeWidth: brushSize / 2,
        points: [scaledCursorPosition.x, scaledCursorPosition.y],
      })
    );
  }, [dispatch, brushSize, tool]);

  const handleMouseMove = useCallback(() => {
    if (!stageRef.current) return;

    const scaledCursorPosition = getScaledCursorPosition(stageRef.current);

    if (!scaledCursorPosition) return;

    dispatch(setCursorPosition(scaledCursorPosition));

    if (!maskLayerRef.current) {
      return;
    }

    const deltaX = lastCursorPosition.current.x - scaledCursorPosition.x;
    const deltaY = lastCursorPosition.current.y - scaledCursorPosition.y;

    lastCursorPosition.current = scaledCursorPosition;

    if (isMovingBoundingBox) {
      const x = _.clamp(
        Math.floor(boundingBoxCoordinate.x - deltaX),
        0,
        canvasDimensions.width - boundingBoxDimensions.width
      );

      const y = _.clamp(
        Math.floor(boundingBoxCoordinate.y - deltaY),
        0,
        canvasDimensions.height - boundingBoxDimensions.height
      );

      dispatch(
        setBoundingBoxCoordinate({
          x,
          y,
        })
      );

      return;
    }
    if (!isDrawing) return;

    didMouseMoveRef.current = true;
    // Extend the current line
    dispatch(
      addPointToCurrentLine([scaledCursorPosition.x, scaledCursorPosition.y])
    );
  }, [
    dispatch,
    isMovingBoundingBox,
    boundingBoxDimensions,
    canvasDimensions,
    boundingBoxCoordinate,
    isDrawing,
  ]);

  const handleMouseUp = useCallback(() => {
    if (!didMouseMoveRef.current && isDrawing && stageRef.current) {
      const scaledCursorPosition = getScaledCursorPosition(stageRef.current);

      if (!scaledCursorPosition || !maskLayerRef.current) return;

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
  }, [dispatch, isDrawing]);

  const handleMouseOutCanvas = useCallback(() => {
    dispatch(setCursorPosition(null));
    dispatch(setIsMovingBoundingBox(false));
    dispatch(setIsDrawing(false));
  }, [dispatch]);

  const handleMouseEnter = useCallback(
    (e: KonvaEventObject<MouseEvent>) => {
      if (e.evt.buttons === 1) {
        if (!stageRef.current) return;

        const scaledCursorPosition = getScaledCursorPosition(stageRef.current);

        if (
          !scaledCursorPosition ||
          !maskLayerRef.current ||
          isMovingBoundingBox ||
          isBoundingBoxTransforming
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
    [dispatch, brushSize, tool, isMovingBoundingBox, isBoundingBoxTransforming]
  );

  return (
    <div className="inpainting-canvas-wrapper checkerboard" tabIndex={1}>
      <div className="inpainting-alerts">
        {!shouldShowMask && <div>Mask Hidden (H)</div>}
        {shouldInvertMask && <div>Mask Inverted (Shift+M)</div>}
      </div>

      {canvasBgImage && (
        <Stage
          width={Math.floor(canvasBgImage.width * stageScale)}
          height={Math.floor(canvasBgImage.height * stageScale)}
          scale={{ x: stageScale, y: stageScale }}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseEnter={handleMouseEnter}
          onMouseUp={handleMouseUp}
          onMouseOut={handleMouseOutCanvas}
          onMouseLeave={handleMouseOutCanvas}
          style={{ cursor: shouldShowMask ? 'none' : 'default' }}
          className="inpainting-canvas-stage"
          ref={stageRef}
        >
          {!shouldInvertMask && !shouldShowCheckboardTransparency && (
            <Layer name={'image-layer'} listening={false}>
              <KonvaImage listening={false} image={canvasBgImage} />
            </Layer>
          )}
          {shouldShowMask && (
            <>
              <Layer
                name={'mask-layer'}
                listening={false}
                opacity={
                  shouldShowCheckboardTransparency || shouldInvertMask
                    ? 1
                    : maskColor.a
                }
                ref={maskLayerRef}
              >
                <InpaintingCanvasLines />
                <InpaintingCanvasBrushPreview />

                {shouldInvertMask && (
                  <KonvaImage
                    image={canvasBgImage}
                    listening={false}
                    globalCompositeOperation="source-in"
                  />
                )}
                {!shouldInvertMask && shouldShowCheckboardTransparency && (
                  <KonvaImage
                    image={canvasBgImage}
                    listening={false}
                    globalCompositeOperation="source-out"
                  />
                )}
              </Layer>
              <Layer>
                {shouldShowBoundingBoxFill && (
                  <InpaintingBoundingBoxPreviewOverlay />
                )}
                <InpaintingBoundingBoxPreview />
                <InpaintingCanvasBrushPreviewOutline />
              </Layer>
            </>
          )}
        </Stage>
      )}
      <Cacher />
      <KeyboardEventManager />
    </div>
  );
};

export default InpaintingCanvas;
