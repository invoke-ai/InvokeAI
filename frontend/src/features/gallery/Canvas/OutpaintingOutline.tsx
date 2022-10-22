import { useState, useEffect } from "react";
import { RootState, useAppSelector } from "../../../app/store";
import { CanvasElementProps } from "./PaintingCanvas";

function OutpaintingOutline(props: CanvasElementProps) {
  const { setOnDraw, unsetOnDraw } = props;

  const { canvasWidth, canvasHeight } = useAppSelector(
    (state: RootState) => state.options,
  );

  useEffect(() => {
    if (setOnDraw && unsetOnDraw) {
      setOnDraw(({ ctx }) => {
        ctx.globalCompositeOperation = "source-over";

        ctx.fillStyle = "rgba(0, 0, 0, 0.5)";
        ctx.fillRect(0, 0, canvasWidth + 4, 2);
        ctx.fillRect(0, 0, 2, canvasHeight + 4);
        ctx.fillRect(canvasWidth + 2, 0, 2, canvasHeight + 4);
        ctx.fillRect(0, canvasHeight + 2, canvasWidth + 4, 2);
      }, -1);

      return () => {
        unsetOnDraw();
      };
    }
  }, [setOnDraw, unsetOnDraw, canvasWidth, canvasHeight]);

  return <></>;
}

export default OutpaintingOutline;