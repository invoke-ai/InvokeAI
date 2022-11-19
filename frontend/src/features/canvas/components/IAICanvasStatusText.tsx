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

    const { cursorX, cursorY } = cursorPosition
      ? { cursorX: cursorPosition.x, cursorY: cursorPosition.y }
      : { cursorX: -1, cursorY: -1 };

    return {
      activeLayerColor:
        layer === 'mask' ? 'var(--status-working-color)' : 'inherit',
      activeLayerString: layer.charAt(0).toUpperCase() + layer.slice(1),
      boundingBoxColor:
        boxWidth < 512 || boxHeight < 512
          ? 'var(--status-working-color)'
          : 'inherit',
      boundingBoxCoordinatesString: `(${roundToHundreth(
        boxX
      )}, ${roundToHundreth(boxY)})`,
      boundingBoxDimensionsString: `${boxWidth}×${boxHeight}`,
      canvasCoordinatesString: `${roundToHundreth(stageX)}×${roundToHundreth(
        stageY
      )}`,
      canvasDimensionsString: `${stageWidth}×${stageHeight}`,
      canvasScaleString: Math.round(stageScale * 100),
      cursorCoordinatesString: `(${cursorX}, ${cursorY})`,
      shouldShowCanvasDebugInfo,
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
    activeLayerColor,
    activeLayerString,
    boundingBoxColor,
    boundingBoxCoordinatesString,
    boundingBoxDimensionsString,
    canvasCoordinatesString,
    canvasDimensionsString,
    canvasScaleString,
    cursorCoordinatesString,
    shouldShowCanvasDebugInfo,
  } = useAppSelector(selector);

  return (
    <div className="canvas-status-text">
      <div
        style={{
          color: activeLayerColor,
        }}
      >{`Active Layer: ${activeLayerString}`}</div>
      <div>{`Canvas Scale: ${canvasScaleString}%`}</div>
      <div
        style={{
          color: boundingBoxColor,
        }}
      >{`Bounding Box: ${boundingBoxDimensionsString}`}</div>
      {shouldShowCanvasDebugInfo && (
        <>
          <div>{`Bounding Box Position: ${boundingBoxCoordinatesString}`}</div>
          <div>{`Canvas Dimensions: ${canvasDimensionsString}`}</div>
          <div>{`Canvas Position: ${canvasCoordinatesString}`}</div>
          <div>{`Cursor Position: ${cursorCoordinatesString}`}</div>
        </>
      )}
    </div>
  );
};

export default IAICanvasStatusText;
