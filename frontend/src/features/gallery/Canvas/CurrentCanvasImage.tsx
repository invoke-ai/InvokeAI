import { useState, useEffect } from "react";
import { RootState, useAppSelector } from "../../../app/store";
import { CanvasElementProps } from "./PaintingCanvas";

import * as InvokeAI from '../../../app/invokeai';

function CurrentCanvasImage(props: CanvasElementProps & { x?: number, y?: number }) {
  const { setOnDraw, unsetOnDraw } = props;

  const currentImage = useAppSelector(
    (state: RootState) => state.gallery.currentImage,
    (a: InvokeAI.Image | undefined, b: InvokeAI.Image | undefined) =>
      a !== undefined && b !== undefined && a.uuid === b.uuid
  );

  const [image, setImage] = useState<HTMLImageElement | null>(null);

  useEffect(() => {
    if (currentImage) {
      const img = new Image();

      img.onload = () => {
        setImage(img);
      };

      img.src = currentImage.url;
    }
  }, [currentImage]);

  useEffect(() => {
    if (setOnDraw && unsetOnDraw) {
      setOnDraw(({ ctx }) => {
        if (image) {
          ctx.drawImage(image, props.x || 0, props.y || 0);
        }
      });

      return () => {
        unsetOnDraw();
      };
    }
  }, [image, setOnDraw, unsetOnDraw, props.x, props.y]);

  return <></>;
}

export default CurrentCanvasImage
