import { createSelector } from '@reduxjs/toolkit';
import _ from 'lodash';
import { RootState } from '../../../app/store';
import { activeTabNameSelector } from '../../options/optionsSelectors';
import { OptionsState } from '../../options/optionsSlice';
import { tabMap } from '../InvokeTabs';
import { InpaintingState } from './inpaintingSlice';
import { rgbaColorToRgbString } from './util/colorToString';

export const inpaintingCanvasLinesSelector = createSelector(
  (state: RootState) => state.inpainting,
  (inpainting: InpaintingState) => {
    const { lines, maskColor } = inpainting;
    return {
      lines,
      maskColorString: rgbaColorToRgbString(maskColor),
    };
  }
);

export const inpaintingControlsSelector = createSelector(
  [
    (state: RootState) => state.inpainting,
    (state: RootState) => state.options,
    activeTabNameSelector,
  ],
  (inpainting: InpaintingState, options: OptionsState, activeTabName) => {
    const {
      tool,
      brushSize,
      maskColor,
      shouldInvertMask,
      shouldShowMask,
      shouldShowCheckboardTransparency,
      lines,
      pastLines,
      futureLines,
      shouldShowBoundingBoxFill,
      shouldShowBoundingBox,
    } = inpainting;

    const { showDualDisplay } = options;

    return {
      tool,
      brushSize,
      maskColor,
      shouldInvertMask,
      shouldShowMask,
      shouldShowCheckboardTransparency,
      canUndo: pastLines.length > 0,
      canRedo: futureLines.length > 0,
      isMaskEmpty: lines.length === 0,
      activeTabName,
      showDualDisplay,
      shouldShowBoundingBoxFill,
      shouldShowBoundingBox,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: _.isEqual,
    },
  }
);

export const inpaintingCanvasSelector = createSelector(
  (state: RootState) => state.inpainting,
  (inpainting: InpaintingState) => {
    const {
      tool,
      brushSize,
      maskColor,
      shouldInvertMask,
      shouldShowMask,
      shouldShowCheckboardTransparency,
      imageToInpaint,
      stageScale,
      shouldShowBoundingBoxFill,
      isDrawing,
      shouldLockBoundingBox,
      shouldShowBoundingBox,
      boundingBoxDimensions,
    } = inpainting;
    return {
      tool,
      brushSize,
      shouldInvertMask,
      shouldShowMask,
      shouldShowCheckboardTransparency,
      maskColor,
      imageToInpaint,
      stageScale,
      shouldShowBoundingBoxFill,
      isDrawing,
      shouldLockBoundingBox,
      shouldShowBoundingBox,
      boundingBoxDimensions,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: (a, b) => {
        const { imageToInpaint: a_imageToInpaint, ...a_rest } = a;
        const { imageToInpaint: b_imageToInpaint, ...b_rest } = b;
        return (
          _.isEqual(a_rest, b_rest) && a_imageToInpaint == b_imageToInpaint
        );
      },
    },
  }
);
