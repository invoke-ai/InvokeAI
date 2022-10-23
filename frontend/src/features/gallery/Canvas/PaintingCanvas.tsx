import React, { MutableRefObject, useEffect, useLayoutEffect, useRef, useState } from "react";
import { createSelector } from "@reduxjs/toolkit";
import { isEqual } from "lodash";
import { RootState, useAppDispatch, useAppSelector } from "../../../app/store";
import { OptionsState } from "../../options/optionsSlice";
import { tabMap } from "../../tabs/InvokeTabs";
import { setPaintingCameraX, setPaintingCameraY, setPaintingElementHeight, setPaintingElementWidth } from "../gallerySlice";
import { isHotkeyPressed } from "react-hotkeys-hook";

interface Point {
  x: number;
  y: number;
}

export interface CanvasElementProps {
  setOnDraw?: (onDraw: (ctx: DrawProps) => void, zIndex: number) => void;
  unsetOnDraw?: () => void;
}

interface PaintingCanvasProps {
  children?: React.ReactNode;
  setOnBrushClear: (onBrushClear: () => void) => void;
}

export interface DrawProps {
  ctx: CanvasRenderingContext2D;
  cameraOffset: Point;
  zoomLevel: number;
  elementWidth: number;
  elementHeight: number;
}

export let canvasRef: MutableRefObject<HTMLCanvasElement | null>;
export let canvasContext: MutableRefObject<CanvasRenderingContext2D | null>;

export let maskCanvasRef: MutableRefObject<HTMLCanvasElement | null>;
export let maskCanvasContext: MutableRefObject<CanvasRenderingContext2D | null>;

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

const PaintingCanvas = (props: PaintingCanvasProps) => {
  const { children, setOnBrushClear } = props;
  const { shouldShowGallery, activeTab, inpaintingTool, inpaintingBrushSize, inpaintingBrushShape } = useAppSelector(
    (state: RootState) => state.options
  );

  const dispatch = useAppDispatch();

  canvasRef = useRef<HTMLCanvasElement>(null);
  canvasContext = useRef<CanvasRenderingContext2D>(null);

  const elementCanvasRef = useRef<HTMLCanvasElement>(null);
  const [elementCanvasContext, setElementCanvasContext] = useState<CanvasRenderingContext2D | null>(null);

  maskCanvasRef = useRef<HTMLCanvasElement>(null);
  maskCanvasContext = useRef<CanvasRenderingContext2D>(null);
  const wrapperRef = useRef<HTMLDivElement>(null);

  // holds the requestAnimationFrame() ID value so we can cancel requests on component unmount
  const animationFrameID = useRef<number>(0);
  const childrenOnDraw = useRef<Map<React.ReactElement, { zIndex: number, onDraw: (ctx: DrawProps) => void }>>(new Map());

  const [cameraOffset, setCameraOffset] = useState<Point>({ x: 0, y: 0 });
  const [isDrawing, setIsDrawing] = useState<boolean>(false);
  const [dragStart, setDragStart] = useState<Point>({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [zoomLevel, setZoomLevel] = useState<number>(1);

  const applyTransform = (x: number, y: number) => {
    if (!wrapperRef.current)
      return { x, y }

    return {
      x: (x - wrapperRef.current.offsetWidth / 2.0) / zoomLevel - cameraOffset.x + wrapperRef.current.offsetWidth / 2.0,
      y: (y - wrapperRef.current.offsetHeight / 2.0) / zoomLevel - cameraOffset.y + wrapperRef.current.offsetHeight / 2.0,
    }
  }

  useEffect(() => {
    if (!canvasRef.current)
      canvasRef.current = document.createElement("canvas");

    if (!canvasContext.current && canvasRef.current)
      canvasContext.current = canvasRef.current.getContext("2d");

    if (!elementCanvasContext && elementCanvasRef.current)
      setElementCanvasContext(elementCanvasRef.current.getContext("2d"));

    if (!maskCanvasRef.current)
      maskCanvasRef.current = document.createElement("canvas");

    if (!maskCanvasContext.current && maskCanvasRef.current)
      maskCanvasContext.current = maskCanvasRef.current.getContext("2d");

  }, [maskCanvasRef, canvasRef.current]);

  useEffect(() => {
    if (!maskCanvasRef.current || !maskCanvasContext.current)
      return;

    setOnBrushClear(() => {
      return () => {
        maskCanvasContext.current!.clearRect(0, 0, maskCanvasRef.current!.width, maskCanvasRef.current!.height);
        draw();
      }
    });
  }, [maskCanvasRef, maskCanvasRef, setOnBrushClear]);

  useEffect(() => {
    if (!maskCanvasRef.current) return;

    maskCanvasRef.current.width = 2048;
    maskCanvasRef.current.height = 2048;
  }, [maskCanvasRef, wrapperRef.current, shouldShowGallery, canvasRef.current]);

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement, MouseEvent>) => {
    if (isHotkeyPressed(" ")) {
      setDragStart({ x: e.clientX / zoomLevel - cameraOffset.x, y: e.clientY / zoomLevel - cameraOffset.y });
      setIsDragging(true);
      return;
    }
    if (e.button === 0) {
      if (
        !maskCanvasRef.current ||
        !maskCanvasContext.current ||
        !wrapperRef.current
      ) return;

      setIsDrawing(true);

      maskCanvasContext.current.fillStyle = "rgba(0, 0, 0, 1)";
      // maskCanvasContext.current.fillStyle = "#000000";

      if (inpaintingTool === "eraser") {
        maskCanvasContext.current.globalCompositeOperation = "destination-out";
      }
      else {
        maskCanvasContext.current.globalCompositeOperation = "source-over";
      }

      const { x, y } = applyTransform(e.clientX - wrapperRef.current.offsetLeft, e.clientY - wrapperRef.current.offsetTop);
      if (inpaintingBrushShape === "circle") {
        maskCanvasContext.current.moveTo(x, y);
        maskCanvasContext.current.arc(x, y, inpaintingBrushSize / 2, 0, 2 * Math.PI, true);
        maskCanvasContext.current.fill();
      }
      else if (inpaintingBrushShape === "square") {
        maskCanvasContext.current.fillRect(x - inpaintingBrushSize / 2, y - inpaintingBrushSize / 2, inpaintingBrushSize, inpaintingBrushSize);
      }

      draw();
    }
  };

  const handleMouseUp = (e: React.MouseEvent<HTMLCanvasElement, MouseEvent>) => {
    if (e.button === 0) {
      setIsDrawing(false);
      setIsDragging(false);
    }
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement, MouseEvent>) => {
    if (!wrapperRef.current) return;

    if (isHotkeyPressed(" ") && isDragging) {
      setCameraOffset({
        x: e.clientX / zoomLevel - dragStart.x,
        y: e.clientY / zoomLevel - dragStart.y,
      });

      draw();
      return;
    }

    if (!maskCanvasRef) return;

    if (isDrawing && maskCanvasContext.current) {
      const { x, y } = applyTransform(e.clientX - wrapperRef.current.offsetLeft, e.clientY - wrapperRef.current.offsetTop);

      if (inpaintingBrushShape === "circle") {
        maskCanvasContext.current.moveTo(x, y);
        maskCanvasContext.current.arc(x, y, inpaintingBrushSize / 2, 0, 2 * Math.PI, true);
        maskCanvasContext.current.fill();
      }
      else if (inpaintingBrushShape === "square") {
        maskCanvasContext.current.fillRect(x - inpaintingBrushSize / 2, y - inpaintingBrushSize / 2, inpaintingBrushSize, inpaintingBrushSize);
      }

      draw();
      return;
    }
  };

  const handleWheel = (e: React.WheelEvent<HTMLCanvasElement>) => {
    if (isHotkeyPressed(" "))
      return;

    if (!wrapperRef.current)
      return;

    const zoomIncrement = 0.1;
    const newZoomLevel = e.deltaY < 0 ? zoomLevel + zoomIncrement : zoomLevel - zoomIncrement;

    if (newZoomLevel < 0.1 || newZoomLevel > 10)
      return;

    setZoomLevel(newZoomLevel);

    const { x, y } = applyTransform(e.clientX - wrapperRef.current.offsetLeft, e.clientY - wrapperRef.current.offsetTop);

    setCameraOffset({
      x: cameraOffset.x - (x - wrapperRef.current.offsetWidth / 2.0) * (e.deltaY < 0 ? zoomIncrement : -zoomIncrement),
      y: cameraOffset.y - (y - wrapperRef.current.offsetHeight / 2.0) * (e.deltaY < 0 ? zoomIncrement : -zoomIncrement),
    });

    draw();
  };

  const draw = () => {
    if (
      !canvasRef.current ||
      !canvasContext.current ||
      !elementCanvasRef.current ||
      !elementCanvasContext ||
      !maskCanvasRef.current ||
      !wrapperRef.current
    ) return;

    const { width, height } = wrapperRef.current.getBoundingClientRect();

    canvasRef.current.width = 4096;
    canvasRef.current.height = 4096;

    elementCanvasRef.current.width = width;
    elementCanvasRef.current.height = height;

    canvasContext.current.globalCompositeOperation = "source-over";
    const childrenDraw = Array
      .from(childrenOnDraw.current.values())
      .concat({
        onDraw: (props) => {
          props.ctx.globalCompositeOperation = "destination-out";
          props.ctx.drawImage(maskCanvasRef.current!, 0, 0);
        }, 
        zIndex: 0
      })
      .sort((a, b) => b.zIndex - a.zIndex)
      .map((child) => child.onDraw);

    childrenDraw.forEach((onDraw) => {
      onDraw({
        ctx: canvasContext.current!,
        elementWidth: width,
        elementHeight: height,
        cameraOffset, zoomLevel
      });
    });

    elementCanvasContext.translate(width / 2, height / 2);
    elementCanvasContext.scale(zoomLevel, zoomLevel);
    elementCanvasContext.translate(-width / 2 + cameraOffset.x, -height / 2 + cameraOffset.y);
    elementCanvasContext.clearRect(0, 0, width, height);
    elementCanvasContext.drawImage(canvasRef.current, 0, 0);

    dispatch(setPaintingCameraX(cameraOffset.x));
    dispatch(setPaintingCameraY(cameraOffset.y));
    dispatch(setPaintingElementWidth(width));
    dispatch(setPaintingElementHeight(height));
  }

  useLayoutEffect(() => {
    draw();

    animationFrameID.current = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(animationFrameID.current);
  }, [cameraOffset, shouldShowGallery, wrapperRef.current]);

  return (
    <div className="painting-canvas" ref={wrapperRef}>
      <canvas className="main-canvas"
        ref={elementCanvasRef}
        onMouseDown={handleMouseDown}
        onMouseUp={handleMouseUp}
        onMouseMove={handleMouseMove}
        onWheel={handleWheel}
      />
      {React.Children.map(children, (child) => {
        if (React.isValidElement(child)) {
          return React.cloneElement<CanvasElementProps>(child as React.FunctionComponentElement<CanvasElementProps>, {
            setOnDraw: (onDraw: (props: DrawProps) => void, zIndex: number) => {
              childrenOnDraw.current.set(child, {
                onDraw: (drawProps) => {
                  onDraw(drawProps);
                },
                zIndex,
              });

              draw();
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