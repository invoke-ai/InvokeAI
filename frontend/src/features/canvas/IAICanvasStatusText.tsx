import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store';
import _ from 'lodash';
import { currentCanvasSelector } from './canvasSlice';

const roundToHundreth = (val: number): number => {
  return Math.round(val * 100) / 100;
};

const selector = createSelector(
  [currentCanvasSelector],
  (currentCanvas) => {
    const {
      stageDimensions: { width: stageWidth, height: stageHeight },
      stageCoordinates: { x: stageX, y: stageY },
      boundingBoxDimensions: { width: boxWidth, height: boxHeight },
      boundingBoxCoordinates: { x: boxX, y: boxY },
      cursorPosition,
      stageScale,
    } = currentCanvas;

    const position = cursorPosition
      ? { cursorX: cursorPosition.x, cursorY: cursorPosition.y }
      : { cursorX: -1, cursorY: -1 };

    return {
      stageWidth,
      stageHeight,
      stageX,
      stageY,
      boxWidth,
      boxHeight,
      boxX,
      boxY,
      stageScale,
      ...position,
    };
  },

  {
    memoizeOptions: {
      resultEqualityCheck: _.isEqual,
    },
  }
);
const IAICanvasStatusText = () => {
  const {
    stageWidth,
    stageHeight,
    stageX,
    stageY,
    boxWidth,
    boxHeight,
    boxX,
    boxY,
    cursorX,
    cursorY,
    stageScale,
  } = useAppSelector(selector);
  return (
    <div className="canvas-status-text">
      <div>{`Stage: ${stageWidth} x ${stageHeight}`}</div>
      <div>{`Stage: ${roundToHundreth(stageX)}, ${roundToHundreth(
        stageY
      )}`}</div>
      <div>{`Scale: ${roundToHundreth(stageScale)}`}</div>
      <div>{`Box: ${boxWidth} x ${boxHeight}`}</div>
      <div>{`Box: ${roundToHundreth(boxX)}, ${roundToHundreth(boxY)}`}</div>
      <div>{`Cursor: ${cursorX}, ${cursorY}`}</div>
    </div>
  );
};

export default IAICanvasStatusText;
