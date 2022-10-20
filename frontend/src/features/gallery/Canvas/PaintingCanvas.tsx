import React, { MutableRefObject, useEffect, useLayoutEffect, useRef, useState } from "react";
import { createSelector } from "@reduxjs/toolkit";
import { isEqual } from "lodash";
import { RootState, useAppSelector } from "../../../app/store";
import { OptionsState } from "../../options/optionsSlice";
import { tabMap } from "../../tabs/InvokeTabs";

import InvokeAI from "../../../app/invokeai";
import drawBrush from "../../tabs/Inpainting/drawBrush";

interface Point {
  x: number;
  y: number;
}

export interface CanvasElementProps {
  setOnDraw?: (onDraw: (ctx: CanvasRenderingContext2D) => void) => void;
  unsetOnDraw?: () => void;
}

export let canvasRef: MutableRefObject<HTMLCanvasElement | null>;

interface PaintingCanvasProps {
  children?: React.ReactNode;
}

const paintingOptionsSelector = createSelector(
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
      shouldShowGallery: options.shouldShowGallery,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: isEqual,
    },
  }
);

const PaintingCanvas = ({ children }: PaintingCanvasProps) => {
  const { shouldShowGallery, activeTab, inpaintingTool, inpaintingBrushSize, inpaintingBrushShape } = useAppSelector(
    (state: RootState) => state.options
  );

  canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapperRef = useRef<HTMLDivElement>(null);
  
  // holds the requestAnimationFrame() ID value so we can cancel requests on component unmount
  const animationFrameID = useRef<number>(0);
  const childrenOnDraw = useRef<Map<React.ReactElement, (ctx: CanvasRenderingContext2D) => void>>(new Map());
  
  const [cameraOffset, setCameraOffset] = useState<Point>({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [isDrawing, setIsDrawing] = useState<boolean>(false);
  const [dragStart, setDragStart] = useState<Point>({ x: 0, y: 0 });
  const [zoomLevel, setZoomLevel] = useState<number>(1);
  const [effectTrigger, setEffectTrigger] = useState<boolean>(false);
  const [maskCanvas, setMaskCanvas] = useState<HTMLCanvasElement | null>(null);
  const [maskContext, setMaskContext] = useState<CanvasRenderingContext2D | null>(null);

  const applyTransform = (x: number, y: number) => {
    if (!wrapperRef.current)
      return { x, y }

    return {
      x: (x - wrapperRef.current.offsetWidth / 2.0) / zoomLevel - cameraOffset.x + wrapperRef.current.offsetWidth / 2.0,
      y: (y - wrapperRef.current.offsetHeight / 2.0) / zoomLevel - cameraOffset.y + wrapperRef.current.offsetHeight / 2.0,
    }
  }

  useEffect(() => {
    setMaskCanvas(document.createElement("canvas"));
  }, []);

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement, MouseEvent>) => {
    if (e.button === 1) {
      setIsDragging(true);
      setDragStart({ x: e.clientX / zoomLevel - cameraOffset.x, y: e.clientY / zoomLevel - cameraOffset.y });
    }
    if (e.button === 0) {
      if (!maskCanvas || !wrapperRef.current) return;

      if (!maskContext) {
        const { width, height } = wrapperRef.current.getBoundingClientRect();

        maskCanvas.width = width;
        maskCanvas.height = height;
        setMaskContext(maskCanvas.getContext("2d"));
      }

      if (!maskContext) return;
      
      setIsDrawing(true);
      maskContext.fillStyle = "#000000";

      const { x, y } = applyTransform(e.clientX - wrapperRef.current.offsetLeft, e.clientY - wrapperRef.current.offsetTop);
      if (inpaintingBrushShape === "circle") {
        maskContext.arc(x, y, inpaintingBrushSize / 2, 0, 2 * Math.PI, true);
        maskContext.fill();
      }
      else {
        maskContext.fillRect(x - inpaintingBrushSize / 2, y - inpaintingBrushSize / 2, inpaintingBrushSize, inpaintingBrushSize);
      }
    }
  };

  const handleMouseUp = (e: React.MouseEvent<HTMLCanvasElement, MouseEvent>) => {
    if (e.button === 1) {
      setIsDragging(false);
    }
    if (e.button === 0) {
      setIsDrawing(false);
    }
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement, MouseEvent>) => {
    if (!maskCanvas || !wrapperRef.current) return;

    if (isDrawing && maskContext) {
      const { x, y } = applyTransform(e.clientX - wrapperRef.current.offsetLeft, e.clientY - wrapperRef.current.offsetTop);

      if (inpaintingBrushShape === "circle") {
        maskContext.arc(x, y, inpaintingBrushSize / 2, 0, 2 * Math.PI, true);
        maskContext.fill();
      }
      else {
        maskContext.fillRect(x - inpaintingBrushSize / 2, y - inpaintingBrushSize / 2, inpaintingBrushSize, inpaintingBrushSize);
      }
    }

    if (!isDragging)
      return;

    setCameraOffset({
      x: e.clientX / zoomLevel - dragStart.x,
      y: e.clientY / zoomLevel - dragStart.y,
    });
  };

  const handleWheel = (e: React.WheelEvent<HTMLCanvasElement>) => {
    if (isDragging)
      return;

    const zoomFactor = 1.1;
    const newZoomLevel = e.deltaY < 0 ? zoomLevel * zoomFactor : zoomLevel / zoomFactor;

    setZoomLevel(newZoomLevel);
  };

  const draw = () => {
    if (!canvasRef.current) return;
    const canvasContext = canvasRef.current.getContext("2d");

    if (
      !canvasContext ||
      !wrapperRef.current
    ) return;

    const { width, height } = wrapperRef.current.getBoundingClientRect();

    canvasRef.current.width = width;
    canvasRef.current.height = height;

    canvasContext.translate(width / 2, height / 2);
    canvasContext.scale(zoomLevel, zoomLevel);
    canvasContext.translate(-width / 2 + cameraOffset.x, -height / 2 + cameraOffset.y);
    canvasContext.clearRect(0, 0, width, height);

    childrenOnDraw.current.forEach((onDraw) => {
      onDraw(canvasContext);
    });
    
    if (maskCanvas)
      canvasContext.drawImage(maskCanvas, 0, 0);
  }

  useLayoutEffect(() => {
    draw();

    animationFrameID.current = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(animationFrameID.current);
  }, [cameraOffset, zoomLevel, shouldShowGallery, effectTrigger]);

  return (
    <div className="painting-canvas" ref={wrapperRef}>
      <canvas className="main-canvas"
        ref={canvasRef}
        onMouseDown={handleMouseDown}
        onMouseUp={handleMouseUp}
        onMouseMove={handleMouseMove}
        onWheel={handleWheel}
      />
      {React.Children.map(children, (child) => {
        if (React.isValidElement(child)) {
          return React.cloneElement<CanvasElementProps>(child as React.FunctionComponentElement<CanvasElementProps>, {
            setOnDraw: (onDraw: (ctx: CanvasRenderingContext2D) => void) => {
              childrenOnDraw.current.set(child, (ctx) => {
                setEffectTrigger(!effectTrigger);
                onDraw(ctx);
              });
            },
            unsetOnDraw: () => {
              childrenOnDraw.current.delete(child);
            }
          });
        }
        return null;
      })}
    </div>
  )
}

export default PaintingCanvas;