import { useState, useEffect } from "react";
import { RootState, useAppSelector } from "../../../app/store";
import { CanvasElementProps } from "./PaintingCanvas";

function OutpaintingHighlight(props: CanvasElementProps) {
  

  const { setOnDraw, unsetOnDraw } = props;

  const { width, height } = useAppSelector(
    (state: RootState) => state.options,
  );

  useEffect(() => {
    if (setOnDraw && unsetOnDraw) {
      setOnDraw(({ ctx, cameraOffset, zoomLevel, elementWidth, elementHeight }) => {
        ctx.fillStyle = "#FF0000";
        ctx.fillRect((elementWidth - width) / 2 - cameraOffset.x, (elementHeight - height) / 2 - cameraOffset.y, width, 2);
        ctx.fillRect((elementWidth - width) / 2 - cameraOffset.x, (elementHeight - height) / 2 - cameraOffset.y, 2, height);
        ctx.fillRect((elementWidth + width) / 2 - cameraOffset.x, (elementHeight - height) / 2 - cameraOffset.y, 2, height);
        ctx.fillRect((elementWidth - width) / 2 - cameraOffset.x, (elementHeight + height) / 2 - cameraOffset.y, width, 2);
      });

      return () => {
        unsetOnDraw();
      };
    }
  }, [setOnDraw, unsetOnDraw, width, height]);

  return <></>;
}

export default OutpaintingHighlight;