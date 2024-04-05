import type { FillRectObject, LayerObject, LineObject } from 'features/regionalPrompts/store/regionalPromptsSlice';
import type { Image } from 'konva/lib/shapes/Image';
import type { Line } from 'konva/lib/shapes/Line';
import type { Rect } from 'konva/lib/shapes/Rect';
import type { Transformer } from 'konva/lib/shapes/Transformer';
import { useEffect, useRef } from 'react';

type ShapeType<T> = T extends LineObject ? Line : T extends FillRectObject ? Rect : Image;

export const useTransform = <TObject extends LayerObject>(object: TObject) => {
  const shapeRef = useRef<ShapeType<TObject>>(null);
  const transformerRef = useRef<Transformer>(null);

  useEffect(() => {
    if (!object.isSelected) {
      return;
    }

    if (!transformerRef.current || !shapeRef.current) {
      return;
    }

    if (object.isSelected) {
      transformerRef.current.nodes([shapeRef.current]);
      transformerRef.current.getLayer()?.batchDraw();
    }
  }, [object.isSelected]);

  return { shapeRef, transformerRef };
};
