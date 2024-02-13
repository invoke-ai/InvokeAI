import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { selectCanvasSlice } from 'features/canvas/store/canvasSlice';
import { rgbaColorToString } from 'features/canvas/util/colorToString';
import { getColoredMaskSVG } from 'features/canvas/util/getColoredMaskSVG';
import type Konva from 'konva';
import type { RectConfig } from 'konva/lib/shapes/Rect';
import { isNumber } from 'lodash-es';
import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Rect } from 'react-konva';

export const canvasMaskCompositerSelector = createMemoizedSelector(selectCanvasSlice, (canvas) => {
  return {
    stageCoordinates: canvas.stageCoordinates,
    stageDimensions: canvas.stageDimensions,
  };
});

type IAICanvasMaskCompositerProps = RectConfig;

const IAICanvasMaskCompositer = (props: IAICanvasMaskCompositerProps) => {
  const { ...rest } = props;

  const { stageCoordinates, stageDimensions } = useAppSelector(canvasMaskCompositerSelector);
  const stageScale = useAppSelector((s) => s.canvas.stageScale);
  const maskColorString = useAppSelector((s) => rgbaColorToString(s.canvas.maskColor));
  const [fillPatternImage, setFillPatternImage] = useState<HTMLImageElement | null>(null);

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
    image.src = getColoredMaskSVG(maskColorString);
  }, [fillPatternImage, maskColorString]);

  useEffect(() => {
    if (!fillPatternImage) {
      return;
    }
    fillPatternImage.src = getColoredMaskSVG(maskColorString);
  }, [fillPatternImage, maskColorString]);

  useEffect(() => {
    const timer = setInterval(() => setOffset((i) => (i + 1) % 5), 50);
    return () => clearInterval(timer);
  }, []);

  const fillPatternScale = useMemo(() => ({ x: 1 / stageScale, y: 1 / stageScale }), [stageScale]);

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
      fillPatternScale={fillPatternScale}
      listening={true}
      globalCompositeOperation="source-in"
      {...rest}
    />
  );
};

export default memo(IAICanvasMaskCompositer);
