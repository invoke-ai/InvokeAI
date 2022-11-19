import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store';
import _ from 'lodash';
import { canvasSelector } from 'features/canvas/store/canvasSelectors';

const roundToHundreth = (val: number): number => {
  return Math.round(val * 100) / 100;
};

const selector = createSelector(
  [canvasSelector],
  (canvas) => {
    const {
      stageDimensions: { width: stageWidth, height: stageHeight },
      stageCoordinates: { x: stageX, y: stageY },
      boundingBoxDimensions: { width: boxWidth, height: boxHeight },
      boundingBoxCoordinates: { x: boxX, y: boxY },
      cursorPosition,
      stageScale,
      shouldShowCanvasDebugInfo,
      layer,
    } = canvas;

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
      shouldShowCanvasDebugInfo,
      layerFormatted: layer.charAt(0).toUpperCase() + layer.slice(1),
      layer,
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
    shouldShowCanvasDebugInfo,
    layer,
    layerFormatted,
  } = useAppSelector(selector);

  return (
    <div className="canvas-status-text">
      <div
        style={{
          color: layer === 'mask' ? 'var(--status-working-color)' : 'inherit',
        }}
      >{`Active Layer: ${layerFormatted}`}</div>
      <div>{`Canvas Scale: ${Math.round(stageScale * 100)}%`}</div>
      <div
        style={{
          color:
            boxWidth < 512 || boxHeight < 512
              ? 'var(--status-working-color)'
              : 'inherit',
        }}
      >{`Bounding Box: ${boxWidth}×${boxHeight}`}</div>
      {shouldShowCanvasDebugInfo && (
        <>
          <div>{`Bounding Box Position: (${roundToHundreth(
            boxX
          )}, ${roundToHundreth(boxY)})`}</div>
          <div>{`Canvas Dimensions: ${stageWidth}×${stageHeight}`}</div>
          <div>{`Canvas Position: ${roundToHundreth(stageX)}×${roundToHundreth(
            stageY
          )}`}</div>
          <div>{`Cursor Position: (${cursorX}, ${cursorY})`}</div>
        </>
      )}
    </div>
  );
};

export default IAICanvasStatusText;
