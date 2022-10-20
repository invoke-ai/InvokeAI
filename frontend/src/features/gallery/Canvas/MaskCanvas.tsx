import { createSelector } from "@reduxjs/toolkit";
import { isEqual } from "lodash";
import { RootState, useAppSelector } from "../../../app/store";
import { OptionsState } from "../../options/optionsSlice";
import { tabMap } from "../../tabs/InvokeTabs";
import { CanvasElementProps } from "./PaintingCanvas";

import InvokeAI from "../../../app/invokeai";

const maskCanvas = document.createElement("canvas");
const maskCanvasContext = maskCanvas.getContext("2d");

export const inpaintingOptionsSelector = createSelector(
  (state: RootState) => state.options,
  (options: OptionsState) => {
    return {
      tool: options.inpaintingTool,
      brushSize: options.inpaintingBrushSize,
      brushShape: options.inpaintingBrushShape,
      // this seems to be a reasonable calculation to get a good brush stamp pixel distance
      brushIncrement: Math.floor(
        Math.min(Math.max(options.inpaintingBrushSize / 8, 1), 5)
      ),
      activeTab: tabMap[options.activeTab],
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: isEqual,
    },
  }
);

const MaskCanvas = (props: CanvasElementProps) => {
  const { setOnDraw, unsetOnDraw } = props;

  const { tool, brushSize, brushShape, brushIncrement } = useAppSelector(
    inpaintingOptionsSelector
  );
  
  
  return <></>
}

export default MaskCanvas