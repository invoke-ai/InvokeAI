import type { Group as KonvaGroupType } from 'konva/lib/Group';
import type { Transformer as KonvaTransformerType } from 'konva/lib/shapes/Transformer';
import { useEffect, useRef } from 'react';

export const useTransform = () => {
  const shapeRef = useRef<KonvaGroupType>(null);
  const transformerRef = useRef<KonvaTransformerType>(null);

  useEffect(() => {
    if (!transformerRef.current || !shapeRef.current) {
      return;
    }
    transformerRef.current.nodes([shapeRef.current]);
    transformerRef.current.getLayer()?.batchDraw();
  }, []);

  return { shapeRef, transformerRef };
};
