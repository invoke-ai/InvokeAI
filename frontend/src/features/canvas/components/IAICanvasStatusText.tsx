import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store';
import _ from 'lodash';
import { canvasSelector } from 'features/canvas/store/canvasSelectors';
import IAICanvasStatusTextCursorPos from './IAICanvasStatusText/IAICanvasStatusTextCursorPos';
import roundToHundreth from '../util/roundToHundreth';

const selector = createSelector(
  [canvasSelector],
  (canvas) => {
    const {
      stageDimensions: { width: stageWidth, height: stageHeight },
      stageCoordinates: { x: stageX, y: stageY },
      boundingBoxDimensions: { width: boxWidth, height: boxHeight },
      boundingBoxCoordinates: { x: boxX, y: boxY },
      stageScale,
      shouldShowCanvasDebugInfo,
      layer,
    } = canvas;

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
          <IAICanvasStatusTextCursorPos />
        </>
      )}
    </div>
  );
};

export default IAICanvasStatusText;
