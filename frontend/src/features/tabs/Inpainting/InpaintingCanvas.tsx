/**
 * Outline of canvas drawing:
 * - useLayoutEffect used to initially set up the canvas
 * - another useLayoutEffect call requestAnimationFrame to kick off the drawing loop
 * - canvas mouse event handlers only set isDrawing / shouldShowBrushPreview booleans, and update cursor position,
 *   all drawing takes place in the drawing loop
 * - Drawing uses multiple offscreen canvases to draw everything, and finally draws the finished offscreen
 *   canvas to the "physical" canvas. This is done to minimize on-screen drawing, which seems to be costly. Cavnases:
 *      - maskCanvas: this is where the mask itself is drawn, it is rendered with globalCompositeOperation 'destination-out',
 *        which makes the image transparent where the maskCanvas has a fill
 *      - brushPreviewCanvas: this is where the cursor / brush preview is drawn (mostly)
 *      - tempCanvas: this is the "final" offscreen canvas, where the above two canvases are drawn. also, the marching ants
 *        brush preview circle is drawn here. I'm undecided if showing it is useful really...
 *      - canvasRef: this is the "physical" canvas, where tempCanvas is drawn, and what the user sees
 * - drawing of the brush is extracted to another function drawBrush, in which we can define brush shapes
 * - The canvas has transparency and its parent div has a classic transparency checkerboard pattern
 * - refs are used instead of useState throughout in an effort to prevent React from doing anything while drawing, we
 *   definitely do not want react to re-render on mousemove etc.
 *
 * Why not Konva? The Konva react bindings encourage (force?) using vectors for lines/shapes, rendering them in the React
 * way, by mapping an array of lines/shapes to logical components which then do the actual drawing.
 *
 * This caused noticeable performance lag when you have many lines/shapes.
 *
 * Another issue is that erasing with Konva's line drawing is not removing portions from existing lines, but
 * instead layering additional lines on with a different globalCompositeOperation. The result is that you may think
 * you have erased the mask by using the eraser, but actually you still have tons of lines being rendered every
 * frame. If we can figure out how to empty the arrays of lines when the mask is fully transparent, then maybe
 * the Konva react bindings way would work.
 *
 * Another option is to use the non-react Konva API to handle drawing. It has a lot of convenient abstractions,
 * I think especially handling an infinite canvas will be easier as a lot of the math is done for you. I haven't
 * investigated this yet.
 *
 * Issues:
 * - Different globalCompositeOperations seem to have different performance characteristics, makes sense as they are diff math
 *   ops. For example using the un-eraser has more lag than the eraser. Not sure there is a way around this...
 */
import { createSelector } from '@reduxjs/toolkit';
import { isEqual } from 'lodash';

import {
  MouseEvent,
  MutableRefObject,
  useLayoutEffect,
  useRef,
  useState,
} from 'react';
import { RootState, useAppDispatch, useAppSelector } from '../../../app/store';
import { OptionsState, setHeight, setWidth } from '../../options/optionsSlice';

import * as InvokeAI from '../../../app/invokeai';

import drawBrush from './drawBrush';
import { tabMap } from '../InvokeTabs';

type Point = {
  x: number;
  y: number;
};

const distanceBetween = (point1: Point, point2: Point) => {
  return Math.sqrt(
    Math.pow(point2.x - point1.x, 2) + Math.pow(point2.y - point1.y, 2)
  );
};

const angleBetween = (point1: Point, point2: Point) => {
  return Math.atan2(point2.x - point1.x, point2.y - point1.y);
};

// // this is only used for the alt line drawing method
// function midPointBtw(p1: Point, p2: Point) {
//   return {
//     x: p1.x + (p2.x - p1.x) / 2,
//     y: p1.y + (p2.y - p1.y) / 2,
//   };
// }

// this stuff should be in a hook

// this allows InvokeButton to get the base64 image data
// must be a `let` bc we reassign it to the ref in this component
export let canvasRef: MutableRefObject<HTMLCanvasElement | null>;

// these allow Clear Mask button to clear the mask canvas and redraw the image
// also prevents rerenders from recreating the mask/brush preview canvases

// must be a `let` bc we reassign it to the ref in this component
export let canvasBgImage: MutableRefObject<HTMLImageElement | null>;

export const maskCanvas = document.createElement('canvas');
export const maskCanvasContext = maskCanvas.getContext('2d');

const brushPreviewCanvas = document.createElement('canvas');
const brushPreviewCanvasContext = brushPreviewCanvas.getContext('2d');

const tempCanvas = document.createElement('canvas');
const tempCanvasContext = tempCanvas.getContext('2d');

export const inpaintingOptionsSelector = createSelector(
  (state: RootState) => state.options,
  (options: OptionsState) => {
    return {
      tool: options.inpaintingTool,
      brushSize: options.inpaintingBrushSize,
      brushShape: options.inpaintingBrushShape,
      // this seems to be a reasonable calculation to get a good brush stamp pixel distance
      brushIncrement: Math.floor(
        Math.min(Math.max(options.inpaintingBrushSize / 8, 1), 5)
      ),
      activeTab: tabMap[options.activeTab],
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

  const { tool, brushSize, brushShape, brushIncrement } = useAppSelector(
    inpaintingOptionsSelector
  );

  const dispatch = useAppDispatch();

  // // this is only used for the alt line drawing method
  // const linePoints = useRef<Point[]>([{ x: 0, y: 0 }]);

  // set the exported canvasRef const to a ref
  canvasRef = useRef<HTMLCanvasElement>(null);

  // set the exported canvasBgImage const to a ref
  canvasBgImage = useRef<HTMLImageElement>(null);

  // used to determine if the canvas should draw the brush preview, set to false when mouse leaves canvas
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

  // container <div> for the <canvas>
  const stageRef = useRef<HTMLDivElement>(null);

  // holds the requestAnimationFrame() ID value so we can cancel requests on component unmount
  const animationFrameID = useRef<number>(0);

  // holds the current image as HTMLImageElement
  const [canvasBgImageLocal, setCanvasBgImageLocal] =
    useState<HTMLImageElement | null>(null);

  // set up canvases when currentImage changes
  useLayoutEffect(() => {
    if (currentImage) {
      const image = new Image();

      // handle setting up canvases
      image.onload = () => {
        if (!canvasRef.current || !stageRef.current) return;

        // the context could be in a ref I guess?
        const canvasContext = canvasRef.current.getContext('2d');

        if (!canvasContext || !tempCanvasContext) return;

        // the w/h of the currentImage
        const { width, height } = image;

        // clean slate for drawing
        canvasContext.clearRect(
          0,
          0,
          canvasRef.current.width,
          canvasRef.current.height
        );

        // this is to get the canvas to fit in the page, this works well but maybe there is a more coherent way

        // get the width of the work area
        const workAreaWidth =
          stageRef.current.parentElement?.parentElement?.clientWidth;

        // and its height
        const workAreaHeight =
          stageRef.current.parentElement?.parentElement?.clientHeight;

        // get the height of the canvas controls UI elements
        const controlsHeight =
          stageRef.current.parentElement?.parentElement?.children[0]
            .clientHeight;

        // have to do this for TS
        if (!workAreaWidth || !workAreaHeight || !controlsHeight) return;

        // calculate the max allowable height of canvas
        const maxHeight = workAreaHeight - controlsHeight;

        // calculate the potential scaling factors of the canvas
        const scaleX = workAreaWidth / width;
        const scaleY = maxHeight / height;

        // set the scaling factor to 1 if the image is small enough to fit, else set it to the min of the above potential scale factors
        const scaleToFit = Math.min(1, Math.min(scaleX, scaleY));

        // calculate the actual pixel of the scaled canvases
        const scaledWidth = Math.floor(scaleToFit * width);
        const scaledHeight = Math.floor(scaleToFit * height);

        // css transform the canvas container div, set its width and height exactly
        stageRef.current.style.transformOrigin = '0 0'; //scale from top left
        stageRef.current.style.transform = `scale(${scaleToFit})`;
        stageRef.current.style.width = `${scaledWidth}px`;
        stageRef.current.style.height = `${scaledHeight}px`;

        // set the physical and offscreen canvases to have the actual pixel widths
        canvasRef.current.width = width;
        canvasRef.current.height = height;
        maskCanvas.width = width;
        maskCanvas.height = height;
        brushPreviewCanvas.width = width;
        brushPreviewCanvas.height = height;
        tempCanvas.width = width;
        tempCanvas.height = height;

        // draw the image
        tempCanvasContext.drawImage(image, 0, 0, width, height);
        canvasContext.drawImage(tempCanvas, 0, 0, width, height);
        canvasBgImage.current = image;

        // set local canvasBgImage to the HTMLImageElement
        setCanvasBgImageLocal(image);

        // set the w/h of the options panel
        dispatch(setWidth(width));
        dispatch(setHeight(height));
      };
      image.src = currentImage.url;
    }
  }, [currentImage, dispatch]);

  useLayoutEffect(() => {
    // // commented to disable marching ants entirely
    // let lineDashOffset = 0;

    if (!canvasRef.current) return;
    const canvasContext = canvasRef.current.getContext('2d');
    // const maskCanvasContext = maskCanvas.getContext('2d');
    // const brushPreviewCanvasContext = brushPreviewCanvas.getContext('2d');

    const draw = () => {
      if (
        !canvasRef.current ||
        !canvasContext ||
        !maskCanvasContext ||
        !brushPreviewCanvasContext ||
        !tempCanvasContext ||
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

        // maskCanvasContext.lineCap = 'round';
        // maskCanvasContext.lineJoin = 'round';
        // maskCanvasContext.lineWidth = brushSize;
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
         * Must fill empty space else the line is just a series of shapes not perfectly overlapping
         * The shape is like the brush - can be anything - even an image
         * This sets us up to have a nice sketchpad
         *
         * In the for loop, brushIncrement can be a greater number for slightly better performance, but at small
         * brush sizes, it needs to be smaller, else the brush isnt smooth any continuous.
         * TODO: calculate in selector
         */

        // calculate distance between cursor positions
        const dist = distanceBetween(
          lastCursorPosition.current,
          currentCursorPosition.current
        );

        // calculate angle between cursor positions
        const angle = angleBetween(
          lastCursorPosition.current,
          currentCursorPosition.current
        );

        // draw the brush shape between the last and current points
        // each brush shape is spaced by `brushIncrement` pixels
        for (let i = 0; i < dist; i += brushIncrement) {
          drawBrush(
            maskCanvasContext,
            brushShape,
            brushSize,
            Math.floor(lastCursorPosition.current.x + Math.sin(angle) * i),
            Math.floor(lastCursorPosition.current.y + Math.cos(angle) * i)
          );
        }
      } else {
        // Here we draw the brush preview, which follows the cursor

        // Reset brush preview canvas - should always start empty as the shape follows cursor and isnt additive
        brushPreviewCanvasContext.clearRect(
          0,
          0,
          canvasRef.current.width,
          canvasRef.current.height
        );

        // Draw and fill in the brush preview shape
        if (shouldShowBrushPreview.current) {
          drawBrush(
            brushPreviewCanvasContext,
            brushShape,
            brushSize,
            currentCursorPosition.current.x,
            currentCursorPosition.current.y
          );
        }
      }

      // Reset canvas for main drawing operations
      canvasContext.clearRect(
        0,
        0,
        canvasRef.current.width,
        canvasRef.current.height
      );

      // and reset the temp canvas
      tempCanvasContext.clearRect(
        0,
        0,
        canvasRef.current.width,
        canvasRef.current.height
      );

      // Draw the image to be masked
      tempCanvasContext.globalCompositeOperation = 'source-over';
      tempCanvasContext.drawImage(canvasBgImageLocal, 0, 0);

      // Draw the mask canvas, compositing out the image
      tempCanvasContext.globalCompositeOperation = 'destination-out';
      tempCanvasContext.drawImage(maskCanvas, 0, 0);

      // Draw the brush preview canvas, compositing it depending on tool selection
      tempCanvasContext.globalCompositeOperation =
        tool === 'eraser' ? 'destination-out' : 'destination-over';
      tempCanvasContext.drawImage(brushPreviewCanvas, 0, 0);

      // Reset the canvas composite so next operations are normal drawing on top of existing image
      tempCanvasContext.globalCompositeOperation = 'source-over';

      // // commented to disable marching ants entirely
      // Draw the brush preview outline w/ marching ants
      // if (shouldShowBrushPreview.current) {
      //   drawBrush(
      //     tempCanvasContext,
      //     brushShape,
      //     brushSize,
      //     lastCursorPosition.current.x,
      //     lastCursorPosition.current.y,
      //     {
      //       lineDashOffset: -lineDashOffset,
      //       lineDash: [4, 2],
      //       strokeStyle: 'black',
      //       lineWidth: 1,
      //     }
      //   );
      //   lineDashOffset += 0.3;
      // }

      canvasContext.drawImage(tempCanvas, 0, 0);

      // update cursor pos
      lastCursorPosition.current = currentCursorPosition.current;

      // update animation frame ID
      animationFrameID.current = requestAnimationFrame(draw);
    };

    draw();
    return () => {
      // when the component is umounted, we need to stop the draw loop
      cancelAnimationFrame(animationFrameID.current);
    };
  }, [tool, brushSize, brushShape, brushIncrement, canvasBgImageLocal]);

  const handleMouseUp = () => {
    isDrawing.current = false;
  };

  const handleMouseDown = (e: MouseEvent<HTMLCanvasElement>) => {
    isDrawing.current = true;

    if (maskCanvasContext) {
      if (tool === 'eraser') {
        maskCanvasContext.globalCompositeOperation = 'source-over';
      } else if (tool === 'uneraser') {
        maskCanvasContext.globalCompositeOperation = 'destination-out';
      }

      // // this is only used for the alt line drawing method
      // linePoints.current = [
      //   { x: e.nativeEvent.offsetX, y: e.nativeEvent.offsetY },
      // ];

      drawBrush(
        maskCanvasContext,
        brushShape,
        brushSize,
        e.nativeEvent.offsetX,
        e.nativeEvent.offsetY
      );
    }
  };

  const handleMouseMove = (e: MouseEvent<HTMLCanvasElement>) => {
    if (!canvasRef.current) return;
    const canvasContext = canvasRef.current.getContext('2d');
    if (!canvasContext) return;
    currentCursorPosition.current = {
      x: e.nativeEvent.offsetX,
      y: e.nativeEvent.offsetY,
    };
    // // this is only used for the alt line drawing method
    // linePoints.current.push({
    //   x: e.nativeEvent.offsetX,
    //   y: e.nativeEvent.offsetY,
    // });
  };

  // the brush preview should never display when mouse is not over canvas
  const handleMouseOutCanvas = () => {
    shouldShowBrushPreview.current = false;
    isDrawing.current = false;
  };

  // the brush preview should always display when mouse is over canvas
  const handleMouseInCanvas = () => {
    shouldShowBrushPreview.current = true;
  };

  return (
    <div className="inpainting-wrapper checkerboard">
      <div ref={stageRef}>
        <canvas
          style={{ cursor: 'none' }}
          ref={canvasRef}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseOut={handleMouseOutCanvas}
          onMouseLeave={handleMouseOutCanvas}
          onMouseOver={handleMouseInCanvas}
          onMouseEnter={handleMouseInCanvas}
        ></canvas>
      </div>
    </div>
  );
};

export default InpaintingCanvas;
