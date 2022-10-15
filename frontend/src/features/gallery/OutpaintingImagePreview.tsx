import { IconButton } from '@chakra-ui/react';
import React, { useState } from 'react';
import { RootState, useAppDispatch, useAppSelector } from '../../app/store';
import * as InvokeAI from '../../app/invokeai';
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
  const { imageToDisplay } = props;
  const dispatch = useAppDispatch();
  
  const [renderTargets, setRenderTargets] = useState<OutpaintingRenderTarget[]>([]);
  const [cameraOffset, setCameraOffset] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [zoomLevel, setZoomLevel] = useState<number>(1);

  const outpaintingCanvasRef = React.useRef<HTMLCanvasElement>(null);

  const render = () => {
    const canvas = outpaintingCanvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.translate(canvas.width / 2, canvas.height / 2);
        ctx.scale(zoomLevel, zoomLevel);
        ctx.translate(-canvas.width / 2 + cameraOffset.x, -canvas.height / 2 + cameraOffset.y);
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        renderTargets.forEach((renderTarget) => {
          ctx.drawImage(renderTarget.target, renderTarget.x, renderTarget.y);
        });
      }
    }
  }

  React.useEffect(() => {
    if (outpaintingCanvasRef.current) {
      const ctx = outpaintingCanvasRef.current.getContext('2d');
      if (ctx) {
        const image = new Image();
        image.src = imageToDisplay.url;

        setRenderTargets([
          {
            target: image,
            x: 0,
            y: 0,
          },
          ...renderTargets,
        ]);

        image.onload = () => render();
      }
    }
  }, [imageToDisplay]);

  // const onOutpaintingCanvasWheel = (event: React.WheelEvent<HTMLCanvasElement>) => {
  //   const canvas = event.currentTarget;
  //   const ctx = canvas.getContext('2d');

  //   if (!ctx)
  //     return;

  //   const scale = event.deltaY > 0 ? 0.9 : 1.1;
  //   const newCanvasScale = canvasScale * scale;

  //   if (newCanvasScale < 0.1 || newCanvasScale > 10)
  //     return;

  //   const rect = canvas.getBoundingClientRect();
  //   const x = event.clientX - rect.left;
  //   const y = event.clientY - rect.top;

  //   ctx.translate(x, y);
  //   ctx.scale(scale, scale);
  //   ctx.translate(-x, -y);

  //   setCanvasScale(newCanvasScale);
  //   console.log(canvasScale, canvasScale);
  // };

  return (
    <div
      className="current-image-preview"
    >
      <canvas
        className="outpainting-canvas"
        style={{
          width: '100%',
          height: '100%',
        }}
        ref={outpaintingCanvasRef}
        // onWheel={onOutpaintingCanvasWheel}
      />
    </div>
  );
}