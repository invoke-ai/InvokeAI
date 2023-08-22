import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { canvasSelector } from 'features/canvas/store/canvasSelectors';
import { RectConfig } from 'konva/lib/shapes/Rect';
import { Rect } from 'react-konva';

import { rgbaColorToString } from 'features/canvas/util/colorToString';
import Konva from 'konva';
import { isNumber } from 'lodash-es';
import { memo, useCallback, useEffect, useRef, useState } from 'react';

export const canvasMaskCompositerSelector = createSelector(
  canvasSelector,
  (canvas) => {
    const { maskColor, stageCoordinates, stageDimensions, stageScale } = canvas;

    return {
      stageCoordinates,
      stageDimensions,
      stageScale,
      maskColorString: rgbaColorToString(maskColor),
    };
  }
);

type IAICanvasMaskCompositerProps = RectConfig;

const getColoredSVG = (color: string) => {
  return `data:image/svg+xml;utf8,<?xml version="1.0" encoding="UTF-8" standalone="no"?>
  <!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
  <svg width="60px" height="60px" viewBox="0 0 30 30" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" xml:space="preserve" xmlns:serif="http://www.serif.com/" style="fill-rule:evenodd;clip-rule:evenodd;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:1.5;">
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

  const { maskColorString, stageCoordinates, stageDimensions, stageScale } =
    useAppSelector(canvasMaskCompositerSelector);

  const [fillPatternImage, setFillPatternImage] =
    useState<HTMLImageElement | null>(null);

  const [offset, setOffset] = useState<number>(0);

  const rectRef = useRef<Konva.Rect>(null);
  const incrementOffset = useCallback(() => {
    setOffset(offset + 1);
    setTimeout(incrementOffset, 500);
  }, [offset]);

  useEffect(() => {
    if (fillPatternImage) {
      return;
    }
    const image = new Image();

    image.onload = () => {
      setFillPatternImage(image);
    };
    image.src = getColoredSVG(maskColorString);
  }, [fillPatternImage, maskColorString]);

  useEffect(() => {
    if (!fillPatternImage) {
      return;
    }
    fillPatternImage.src = getColoredSVG(maskColorString);
  }, [fillPatternImage, maskColorString]);

  useEffect(() => {
    const timer = setInterval(() => setOffset((i) => (i + 1) % 5), 50);
    return () => clearInterval(timer);
  }, []);

  if (
    !fillPatternImage ||
    !isNumber(stageCoordinates.x) ||
    !isNumber(stageCoordinates.y) ||
    !isNumber(stageScale) ||
    !isNumber(stageDimensions.width) ||
    !isNumber(stageDimensions.height)
  ) {
    return null;
  }

  return (
    <Rect
      ref={rectRef}
      offsetX={stageCoordinates.x / stageScale}
      offsetY={stageCoordinates.y / stageScale}
      height={stageDimensions.height / stageScale}
      width={stageDimensions.width / stageScale}
      fillPatternImage={fillPatternImage}
      fillPatternOffsetY={!isNumber(offset) ? 0 : offset}
      fillPatternRepeat="repeat"
      fillPatternScale={{ x: 1 / stageScale, y: 1 / stageScale }}
      listening={true}
      globalCompositeOperation="source-in"
      {...rest}
    />
  );
};

export default memo(IAICanvasMaskCompositer);
