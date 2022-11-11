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
import maskPatternImage from 'assets/images/mask_pattern2.png';

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
    console.log('toggle');
    console.log('incrementing');
  }, [offset]);

  useEffect(() => {
    if (fillPatternImage) return;
    const image = new Image();

    image.onload = () => {
      setFillPatternImage(image);
    };

    image.src = maskPatternImage;
  }, [fillPatternImage]);

  useEffect(() => {
    const timer = setInterval(() => setOffset((i) => (i + 1) % 6), 100);
    return () => clearInterval(timer);
  }, []);

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
      fillPatternScale={{ x: 1 / stageScale, y: 1 / stageScale }}
      listening={true}
      opacity={maskOpacity}
      globalCompositeOperation={'source-in'}
      {...rest}
    />
  );
};

export default IAICanvasMaskCompositer;
