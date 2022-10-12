import {
  Slider,
  SliderTrack,
  SliderFilledTrack,
  SliderThumb,
  Switch,
  Flex,
  FormControl,
  FormLabel,
  SliderMark,
} from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { isEqual } from 'lodash';

import {
  MouseEvent,
  MutableRefObject,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
} from 'react';
import { FaEraser, FaPaintBrush, FaTrash } from 'react-icons/fa';
import { RootState, useAppSelector } from '../../../app/store';
import IAIIconButton from '../../../common/components/IAIIconButton';
import { OptionsState } from '../../options/optionsSlice';

import * as InvokeAI from '../../../app/invokeai';

type Tool = 'pen' | 'eraser';

type Point = {
  x: number;
  y: number;
};

function distanceBetween(point1: Point, point2: Point) {
  return Math.sqrt(
    Math.pow(point2.x - point1.x, 2) + Math.pow(point2.y - point1.y, 2)
  );
}

function angleBetween(point1: Point, point2: Point) {
  return Math.atan2(point2.x - point1.x, point2.y - point1.y);
}

function midPointBtw(p1: Point, p2: Point) {
  return {
    x: p1.x + (p2.x - p1.x) / 2,
    y: p1.y + (p2.y - p1.y) / 2,
  };
}

export let canvasRef: MutableRefObject<HTMLCanvasElement | null>;
export let canvasBgImage: MutableRefObject<HTMLImageElement | null>;
export const maskCanvas = document.createElement('canvas');
const brushPreviewCanvas = document.createElement('canvas');

export const inpaintingOptionsSelector = createSelector(
  (state: RootState) => state.options,
  (options: OptionsState) => {
    return {
      inpaintingTool: options.inpaintingTool,
      inpaintingBrushRadius: options.inpaintingBrushRadius,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: isEqual,
    },
  }
);

const InpaintingCanvas = () => {
  const currentImage = useAppSelector(
    (state: RootState) => state.gallery.currentImage,
    (a: InvokeAI.Image | undefined, b: InvokeAI.Image | undefined) =>
      a !== undefined && b !== undefined && a.uuid === b.uuid
  );

  canvasRef = useRef<HTMLCanvasElement>(null);

  // const [isDrawing, setIsDrawing] = useState<boolean>(false);

  // TODO: add mask invert display (so u can see exactly what parts of image are masked)
  const [shouldInvertMask, setShouldInvertMask] = useState<boolean>(false);

  // TODO: add mask overlay display
  const [shouldOverlayMask, setShouldOverlayMask] = useState<boolean>(false);

  const shouldShowBrushPreview = useRef<boolean>(false);

  const lastCursorPosition = useRef<Point>({
    x: 0,
    y: 0,
  });

  const currentCursorPosition = useRef<Point>({
    x: 0,
    y: 0,
  });

  const isDrawing = useRef<boolean>(false);
  // const [lastCursorPosition, setLastCursorPosition] = useState<Point>({
  //   x: 0,
  //   y: 0,
  // });
  // const [currentCursorPosition, setCurrentCursorPosition] = useState<Point>({
  //   x: 0,
  //   y: 0,
  // });

  const linePoints = useRef<Point[]>([{ x: 0, y: 0 }]);

  const [canvasBgImageLocal, setCanvasBgImageLocal] =
    useState<HTMLImageElement | null>(null);

  const { inpaintingTool: tool, inpaintingBrushRadius: brushRadius } =
    useAppSelector(inpaintingOptionsSelector);

  // TODO: UI for eraser vs pen... the main action is erasing, but then the secondary
  // is like un-erasing... confusing
  // const [tool, setTool] = useState<Tool>('pen')// const [brushRadius, setBrushRadius] = useState<number>(20);

  // const [canvasBgImage, setCanvasBgImage] = useState<HTMLImageElement | null>(
  //   null
  // );

  canvasRef = useRef<HTMLCanvasElement>(null);
  canvasBgImage = useRef<HTMLImageElement>(null);

  useLayoutEffect(() => {
    if (currentImage) {
      const image = new Image();

      image.onload = () => {
        if (!canvasRef.current) return;

        const canvasContext = canvasRef.current.getContext('2d');

        if (!canvasContext) return;
        const { width, height } = image;

        canvasContext.clearRect(
          0,
          0,
          canvasRef.current.width,
          canvasRef.current.height
        );

        canvasRef.current.width = width;
        canvasRef.current.height = height;
        maskCanvas.width = width;
        maskCanvas.height = height;
        brushPreviewCanvas.width = width;
        brushPreviewCanvas.height = height;

        canvasContext.drawImage(image, 0, 0, width, height);
        canvasBgImage.current = image;
        setCanvasBgImageLocal(image);
      };
      image.src = currentImage.url;
    }
  }, [currentImage]);

  const handleMouseDown = (e: MouseEvent<HTMLCanvasElement>) => {
    // const { offsetX: x, offsetY: y } = e.nativeEvent;
    // lastCursorPosition.current =
    // setLastCursorPosition({ x, y });
    isDrawing.current = true;

    linePoints.current = [
      { x: e.nativeEvent.offsetX, y: e.nativeEvent.offsetY },
    ];

    const maskCanvasContext = maskCanvas.getContext('2d');
    if (maskCanvasContext) {
      if (tool === 'eraser') {
        maskCanvasContext.globalCompositeOperation = 'source-over';
      } else if (tool === 'uneraser') {
        maskCanvasContext.globalCompositeOperation = 'destination-out';
      }
      maskCanvasContext.beginPath();
      maskCanvasContext.arc(
        e.nativeEvent.offsetX,
        e.nativeEvent.offsetY,
        brushRadius,
        0,
        Math.PI * 2
      );
      maskCanvasContext.fill();
      maskCanvasContext.closePath();
    }
  };

  const handleMouseMove = (e: MouseEvent<HTMLCanvasElement>) => {
    currentCursorPosition.current = {
      x: e.nativeEvent.offsetX,
      y: e.nativeEvent.offsetY,
    };
    linePoints.current.push({
      x: e.nativeEvent.offsetX,
      y: e.nativeEvent.offsetY,
    });
    // setCurrentCursorPosition({
    //   x: e.nativeEvent.offsetX,
    //   y: e.nativeEvent.offsetY,
    // });
  };

  useLayoutEffect(() => {
    let requestId: number;
    const lineDashOffset = 0;

    if (!canvasRef.current) return;
    const canvasContext = canvasRef.current.getContext('2d');
    const maskCanvasContext = maskCanvas.getContext('2d');
    const brushPreviewCanvasContext = brushPreviewCanvas.getContext('2d');

    const draw = () => {
      if (
        !canvasRef.current ||
        !canvasContext ||
        !maskCanvasContext ||
        !brushPreviewCanvasContext ||
        !canvasBgImageLocal
      )
        return;

      if (isDrawing.current) {
        // here we add to/subtract from the mask

        // 'eraser' adds to the mask, which erases the image
        // 'uneraser' subtracts from the mask, which 'un-erases' the image
        if (tool === 'eraser') {
          maskCanvasContext.globalCompositeOperation = 'source-over';
        } else if (tool === 'uneraser') {
          maskCanvasContext.globalCompositeOperation = 'destination-out';
        }

        // two good methods:

        /**
         * Draw lines between points using quadraticCurveTo():
         * Similar to Konva's implementation of free drawing using lines
         * Can also use canvas's bezier curves?
         * Cannot do brushes - its just stroked lines
         */

        // let p1 = linePoints.current[0];
        // let p2 = linePoints.current[1];

        // maskCanvasContext.lineCap = 'round'
        // maskCanvasContext.lineJoin = 'round'
        // maskCanvasContext.lineWidth = brushRadius * 2;
        // maskCanvasContext.beginPath();
        // maskCanvasContext.moveTo(p1.x, p1.y);

        // for (let i = 1, len = linePoints.current.length; i < len; i++) {
        //   // we pick the point between pi+1 & pi+2 as the
        //   // end point and p1 as our control point
        //   const midPoint = midPointBtw(p1, p2);
        //   maskCanvasContext.quadraticCurveTo(
        //     p1.x,
        //     p1.y,
        //     midPoint.x,
        //     midPoint.y
        //   );
        //   p1 = linePoints.current[i];
        //   p2 = linePoints.current[i + 1];
        // }
        // // Draw last line as a straight line while
        // // we wait for the next point to be able to calculate
        // // the bezier control point
        // maskCanvasContext.lineTo(p1.x, p1.y);
        // maskCanvasContext.stroke();

        /**
         * Draw with individual shapes:
         * Calculate the distance/angle between each point and fill the empty space with the shape
         * The shape is like the brush - can be anything - even an image
         * This sets us up to have a nice sketchpad
         */
        const dist = distanceBetween(
          lastCursorPosition.current,
          currentCursorPosition.current
        );
        const angle = angleBetween(
          lastCursorPosition.current,
          currentCursorPosition.current
        );
        maskCanvasContext.beginPath();
        for (let i = 0; i < dist; i += 5) {
          const x = lastCursorPosition.current.x + Math.sin(angle) * i;
          const y = lastCursorPosition.current.y + Math.cos(angle) * i;
          // circle brush
          maskCanvasContext.arc(x, y, brushRadius, 0, Math.PI * 2);

          // square brush example
          // maskCanvasContext.rect(
          //   x - brushRadius,
          //   y - brushRadius,
          //   brushRadius * 2,
          //   brushRadius * 2
          // );

          maskCanvasContext.fill();
        }
        maskCanvasContext.closePath();
      } else {
        // Here we draw the brush preview, which follows the cursor

        // Reset brush preview canvas - should always start empty as the shape follows cursor
        brushPreviewCanvasContext.clearRect(
          0,
          0,
          canvasRef.current.width,
          canvasRef.current.height
        );

        // Draw and fill in the brush preview shape
        if (shouldShowBrushPreview.current) {
          brushPreviewCanvasContext.beginPath();
          brushPreviewCanvasContext.arc(
            currentCursorPosition.current.x,
            currentCursorPosition.current.y,
            brushRadius,
            0,
            Math.PI * 2
          );
          brushPreviewCanvasContext.fill();
          brushPreviewCanvasContext.closePath();
        }
      }

      // Reset canvas for main drawing operations
      canvasContext.clearRect(
        0,
        0,
        canvasRef.current.width,
        canvasRef.current.height
      );

      // Draw the image to be masked
      canvasContext.globalCompositeOperation = 'source-over';
      canvasContext.drawImage(canvasBgImageLocal, 0, 0);

      // Draw the mask canvas, compositing out the image
      canvasContext.globalCompositeOperation = 'destination-out';
      canvasContext.drawImage(maskCanvas, 0, 0);

      // Draw the brush preview canvas, compositing it depending on tool selection
      canvasContext.globalCompositeOperation =
        tool === 'eraser' ? 'destination-out' : 'destination-over';
      canvasContext.drawImage(brushPreviewCanvas, 0, 0);

      // Reset the canvas composite so next operations are normal drawing on top of existing image
      canvasContext.globalCompositeOperation = 'source-over';

      // Draw the brush preview outline w/ marching ants
      if (shouldShowBrushPreview.current) {
        canvasContext.beginPath();
        canvasContext.arc(
          lastCursorPosition.current.x,
          lastCursorPosition.current.y,
          brushRadius,
          Math.PI * 2,
          0
        );
        canvasContext.lineDashOffset = -lineDashOffset;
        canvasContext.setLineDash([4, 2]);
        canvasContext.strokeStyle = 'black';
        canvasContext.lineWidth = 1;
        canvasContext.stroke();
        canvasContext.closePath();
        // lineDashOffset += 0.3; // Disable Marching Ants
      }

      // update cursor pos
      lastCursorPosition.current = currentCursorPosition.current;

      // update animation frame ID
      requestId = requestAnimationFrame(draw);
    };

    draw();
    return () => cancelAnimationFrame(requestId);
  }, [brushRadius, tool, canvasBgImageLocal, shouldShowBrushPreview]);

  const handleMouseUp = () => {
    isDrawing.current = false;
  };

  const handleMouseOutCanvas = () => {
    shouldShowBrushPreview.current = false;
    isDrawing.current = false;
  };

  const handleMouseInCanvas = () => {
    shouldShowBrushPreview.current = true;
  };

  return (
    <canvas
      ref={canvasRef}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseOut={handleMouseOutCanvas}
      onMouseLeave={handleMouseOutCanvas}
      onMouseOver={handleMouseInCanvas}
      onMouseEnter={handleMouseInCanvas}
    ></canvas>
  );
};

export default InpaintingCanvas;
