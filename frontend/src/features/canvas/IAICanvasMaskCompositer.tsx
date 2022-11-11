import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store';
import { RectConfig } from 'konva/lib/shapes/Rect';
import _ from 'lodash';
import { Rect } from 'react-konva';
import {
  currentCanvasSelector,
  InpaintingCanvasState,
  OutpaintingCanvasState,
} from './canvasSlice';

import { rgbaColorToString } from './util/colorToString';
import { useCallback, useEffect, useRef, useState } from 'react';
import Konva from 'konva';

export const canvasMaskCompositerSelector = createSelector(
  currentCanvasSelector,
  (currentCanvas) => {
    const { maskColor, stageCoordinates, stageDimensions, stageScale } =
      currentCanvas as InpaintingCanvasState | OutpaintingCanvasState;

    return {
      stageCoordinates,
      stageDimensions,
      stageScale,
      maskColorString: rgbaColorToString(maskColor),
      maskOpacity: maskColor.a,
    };
  }
);

type IAICanvasMaskCompositerProps = RectConfig;

const getColoredSVG = (color: string) => {
  return `data:image/svg+xml;utf8,<svg width="100%" height="100%" viewBox="0 0 30 30" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" xml:space="preserve" xmlns:serif="http://www.serif.com/" style="fill-rule:evenodd;clip-rule:evenodd;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:1.5;">
  <g transform="matrix(0.5,0,0,0.5,0,0)">
      <path d="M-3.5,63.5L64,-4" style="fill:none;stroke:black;stroke-width:1px;"/>
  </g>
  <g transform="matrix(0.5,0,0,0.5,0,2.5)">
      <path d="M-3.5,63.5L64,-4" style="fill:none;stroke:black;stroke-width:1px;"/>
  </g>
  <g transform="matrix(0.5,0,0,0.5,0,5)">
      <path d="M-3.5,63.5L64,-4" style="fill:none;stroke:black;stroke-width:1px;"/>
  </g>
  <g transform="matrix(0.5,0,0,0.5,0,7.5)">
      <path d="M-3.5,63.5L64,-4" style="fill:none;stroke:black;stroke-width:1px;"/>
  </g>
  <g transform="matrix(0.5,0,0,0.5,0,10)">
      <path d="M-3.5,63.5L64,-4" style="fill:none;stroke:black;stroke-width:1px;"/>
  </g>
  <g transform="matrix(0.5,0,0,0.5,0,12.5)">
      <path d="M-3.5,63.5L64,-4" style="fill:none;stroke:black;stroke-width:1px;"/>
  </g>
  <g transform="matrix(0.5,0,0,0.5,0,15)">
      <path d="M-3.5,63.5L64,-4" style="fill:none;stroke:black;stroke-width:1px;"/>
  </g>
  <g transform="matrix(0.5,0,0,0.5,0,17.5)">
      <path d="M-3.5,63.5L64,-4" style="fill:none;stroke:black;stroke-width:1px;"/>
  </g>
  <g transform="matrix(0.5,0,0,0.5,0,20)">
      <path d="M-3.5,63.5L64,-4" style="fill:none;stroke:black;stroke-width:1px;"/>
  </g>
  <g transform="matrix(0.5,0,0,0.5,0,22.5)">
      <path d="M-3.5,63.5L64,-4" style="fill:none;stroke:black;stroke-width:1px;"/>
  </g>
  <g transform="matrix(0.5,0,0,0.5,0,25)">
      <path d="M-3.5,63.5L64,-4" style="fill:none;stroke:black;stroke-width:1px;"/>
  </g>
  <g transform="matrix(0.5,0,0,0.5,0,27.5)">
      <path d="M-3.5,63.5L64,-4" style="fill:none;stroke:black;stroke-width:1px;"/>
  </g>
  <g transform="matrix(0.5,0,0,0.5,0,30)">
      <path d="M-3.5,63.5L64,-4" style="fill:none;stroke:black;stroke-width:1px;"/>
  </g>
  <g transform="matrix(0.5,0,0,0.5,0,-2.5)">
      <path d="M-3.5,63.5L64,-4" style="fill:none;stroke:black;stroke-width:1px;"/>
  </g>
  <g transform="matrix(0.5,0,0,0.5,0,-5)">
      <path d="M-3.5,63.5L64,-4" style="fill:none;stroke:black;stroke-width:1px;"/>
  </g>
  <g transform="matrix(0.5,0,0,0.5,0,-7.5)">
      <path d="M-3.5,63.5L64,-4" style="fill:none;stroke:black;stroke-width:1px;"/>
  </g>
  <g transform="matrix(0.5,0,0,0.5,0,-10)">
      <path d="M-3.5,63.5L64,-4" style="fill:none;stroke:black;stroke-width:1px;"/>
  </g>
  <g transform="matrix(0.5,0,0,0.5,0,-12.5)">
      <path d="M-3.5,63.5L64,-4" style="fill:none;stroke:black;stroke-width:1px;"/>
  </g>
  <g transform="matrix(0.5,0,0,0.5,0,-15)">
      <path d="M-3.5,63.5L64,-4" style="fill:none;stroke:black;stroke-width:1px;"/>
  </g>
  <g transform="matrix(0.5,0,0,0.5,0,-17.5)">
      <path d="M-3.5,63.5L64,-4" style="fill:none;stroke:black;stroke-width:1px;"/>
  </g>
  <g transform="matrix(0.5,0,0,0.5,0,-20)">
      <path d="M-3.5,63.5L64,-4" style="fill:none;stroke:black;stroke-width:1px;"/>
  </g>
  <g transform="matrix(0.5,0,0,0.5,0,-22.5)">
      <path d="M-3.5,63.5L64,-4" style="fill:none;stroke:black;stroke-width:1px;"/>
  </g>
  <g transform="matrix(0.5,0,0,0.5,0,-25)">
      <path d="M-3.5,63.5L64,-4" style="fill:none;stroke:black;stroke-width:1px;"/>
  </g>
  <g transform="matrix(0.5,0,0,0.5,0,-27.5)">
      <path d="M-3.5,63.5L64,-4" style="fill:none;stroke:black;stroke-width:1px;"/>
  </g>
  <g transform="matrix(0.5,0,0,0.5,0,-30)">
      <path d="M-3.5,63.5L64,-4" style="fill:none;stroke:black;stroke-width:1px;"/>
  </g>
</svg>`.replaceAll('black', color);
};

const IAICanvasMaskCompositer = (props: IAICanvasMaskCompositerProps) => {
  const { ...rest } = props;

  const {
    maskColorString,
    maskOpacity,
    stageCoordinates,
    stageDimensions,
    stageScale,
  } = useAppSelector(canvasMaskCompositerSelector);

  const [fillPatternImage, setFillPatternImage] =
    useState<HTMLImageElement | null>(null);

  const [offset, setOffset] = useState<number>(0);

  const rectRef = useRef<Konva.Rect>(null);
  const incrementOffset = useCallback(() => {
    setOffset(offset + 1);
    setTimeout(incrementOffset, 500);
  }, [offset]);

  useEffect(() => {
    if (fillPatternImage) return;
    const image = new Image();

    image.onload = () => {
      setFillPatternImage(image);
    };
    image.src = getColoredSVG(maskColorString);
  }, [fillPatternImage, maskColorString]);

  useEffect(() => {
    if (!fillPatternImage) return;
    fillPatternImage.src = getColoredSVG(maskColorString);
  }, [fillPatternImage, maskColorString]);

  useEffect(() => {
    const timer = setInterval(
      () => setOffset((i) => (i + 2) % Number(fillPatternImage?.width)),
      100
    );
    return () => clearInterval(timer);
  }, [fillPatternImage?.width]);

  if (!fillPatternImage) return null;

  return (
    <Rect
      ref={rectRef}
      offsetX={stageCoordinates.x / stageScale}
      offsetY={stageCoordinates.y / stageScale}
      height={stageDimensions.height / stageScale}
      width={stageDimensions.width / stageScale}
      fillPatternImage={fillPatternImage}
      fillPatternOffsetY={offset}
      fillPatternRepeat={'repeat'}
      fillPatternScale={{ x: 0.5 / stageScale, y: 0.5 / stageScale }}
      listening={true}
      opacity={maskOpacity}
      globalCompositeOperation={'source-in'}
      {...rest}
    />
  );
};

export default IAICanvasMaskCompositer;
