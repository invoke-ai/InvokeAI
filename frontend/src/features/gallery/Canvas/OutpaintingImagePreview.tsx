import { IconButton } from '@chakra-ui/react';
import React, { useState } from 'react';
import { RootState, useAppDispatch, useAppSelector } from '../../../app/store';
import * as InvokeAI from '../../../app/invokeai';
import _ from 'lodash';

interface OutpaintingImagePreviewProps {
  imageToDisplay: InvokeAI.Image;
}

interface OutpaintingRenderTarget {
  target: HTMLImageElement;
  x: number;
  y: number;
}

export default function OutpaintingImagePreview(props: OutpaintingImagePreviewProps) {
  const { 
    canvasWidth, 
    canvasHeight,
    width,
    height, 
  } = useAppSelector(
    (state: RootState) => state.options
  )
 
  const { imageToDisplay } = props;
  const dispatch = useAppDispatch();

  const [renderTargets, setRenderTargets] = useState<OutpaintingRenderTarget[]>([{
    target: (() => {
      const image = new Image();
      image.src = imageToDisplay.url;

      return image;
    })(),
    x: 0,
    y: 0
  }]);
  const [cameraOffset, setCameraOffset] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [dragStart, setDragStart] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [zoomLevel, setZoomLevel] = useState<number>(1);

  const outpaintingCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const imagePreviewRef = React.useRef<HTMLDivElement>(null);

  const render = () => {
    const canvas = outpaintingCanvasRef.current;

    if (!canvas)
      return;

    if (imagePreviewRef.current) {
      canvas.width = imagePreviewRef.current.clientWidth;
      canvas.height = imagePreviewRef.current.clientHeight;
    }

    const ctx = canvas.getContext('2d');
    if (!ctx)
      return;

    ctx.translate(canvas.width / 2, canvas.height / 2);
    ctx.scale(zoomLevel, zoomLevel);
    ctx.translate(-canvas.width / 2 + cameraOffset.x, -canvas.height / 2 + cameraOffset.y);

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    renderTargets.forEach((renderTarget) => {
      ctx.drawImage(renderTarget.target, renderTarget.x, renderTarget.y);
    });

    ctx.fillStyle = 'rgba(0, 0, 0, 1)';
    ctx.fillRect(0, -1, canvasWidth, 1);
    ctx.fillRect(-1, 0, 1, canvasHeight);
    ctx.fillRect(canvasWidth, 0, 1, canvasHeight);
    ctx.fillRect(0, canvasHeight, canvasWidth, 1);

    ctx.fillStyle = '#FF0000';
    ctx.fillRect(canvas.width / 2 - cameraOffset.x - width / 2, canvas.height / 2 - cameraOffset.y - height / 2, width, 2);
    ctx.fillRect(canvas.width / 2 - cameraOffset.x - width / 2, canvas.height / 2 - cameraOffset.y - height / 2, 2, height);
    ctx.fillRect(canvas.width / 2 - cameraOffset.x + width / 2, canvas.height / 2 - cameraOffset.y - height / 2, 2, height);
    ctx.fillRect(canvas.width / 2 - cameraOffset.x - width / 2, canvas.height / 2 - cameraOffset.y + height / 2, width, 2);
  }

  React.useEffect(() => {
    render();
  }, [imageToDisplay]);

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement, MouseEvent>) => {
    setIsDragging(true);
    setDragStart({ x: e.clientX / zoomLevel - cameraOffset.x, y: e.clientY / zoomLevel - cameraOffset.y });
  }

  const handleMouseUp = (e: React.MouseEvent<HTMLCanvasElement, MouseEvent>) => {
    setIsDragging(false);
  }

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement, MouseEvent>) => {
    if (!isDragging)
      return;

    setCameraOffset({
      x: e.clientX / zoomLevel - dragStart.x,
      y: e.clientY / zoomLevel - dragStart.y
    });
    render();
  }

  const handleWheel = (e: React.WheelEvent<HTMLCanvasElement>) => {
    if (isDragging)
      return;

    const zoomFactor = 1.1;
    const newZoomLevel = e.deltaY < 0 ? zoomLevel * zoomFactor : zoomLevel / zoomFactor;

    setZoomLevel(newZoomLevel);
    render();
  }

  return (
    <div className="outpainting-image-preview" ref={imagePreviewRef}>
      <canvas
        className="outpainting-canvas"
        ref={outpaintingCanvasRef}
        onMouseDown={handleMouseDown}
        onMouseUp={handleMouseUp}
        onMouseMove={handleMouseMove}
        onWheel={handleWheel}
      />
    </div>
  );
}