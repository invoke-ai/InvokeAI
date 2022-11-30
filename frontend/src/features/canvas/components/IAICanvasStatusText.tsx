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
      scaledBoundingBoxDimensions: {
        width: scaledBoxWidth,
        height: scaledBoxHeight,
      },
      boundingBoxCoordinates: { x: boxX, y: boxY },
      stageScale,
      shouldShowCanvasDebugInfo,
      layer,
      boundingBoxScaleMethod,
    } = canvas;

    let boundingBoxColor = 'inherit';

    if (
      (boundingBoxScaleMethod === 'none' &&
        (boxWidth < 512 || boxHeight < 512)) ||
      (boundingBoxScaleMethod === 'manual' &&
        scaledBoxWidth * scaledBoxHeight < 512 * 512)
    ) {
      boundingBoxColor = 'var(--status-working-color)';
    }

    const activeLayerColor =
      layer === 'mask' ? 'var(--status-working-color)' : 'inherit';

    return {
      activeLayerColor,
      activeLayerString: layer.charAt(0).toUpperCase() + layer.slice(1),
      boundingBoxColor,
      boundingBoxCoordinatesString: `(${roundToHundreth(
        boxX
      )}, ${roundToHundreth(boxY)})`,
      boundingBoxDimensionsString: `${boxWidth}×${boxHeight}`,
      scaledBoundingBoxDimensionsString: `${scaledBoxWidth}×${scaledBoxHeight}`,
      canvasCoordinatesString: `${roundToHundreth(stageX)}×${roundToHundreth(
        stageY
      )}`,
      canvasDimensionsString: `${stageWidth}×${stageHeight}`,
      canvasScaleString: Math.round(stageScale * 100),
      shouldShowCanvasDebugInfo,
      shouldShowBoundingBox: boundingBoxScaleMethod !== 'auto',
      shouldShowScaledBoundingBox: boundingBoxScaleMethod !== 'none',
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
    scaledBoundingBoxDimensionsString,
    shouldShowScaledBoundingBox,
    canvasCoordinatesString,
    canvasDimensionsString,
    canvasScaleString,
    shouldShowCanvasDebugInfo,
    shouldShowBoundingBox,
  } = useAppSelector(selector);

  return (
    <div className="canvas-status-text">
      <div
        style={{
          color: activeLayerColor,
        }}
      >{`Active Layer: ${activeLayerString}`}</div>
      <div>{`Canvas Scale: ${canvasScaleString}%`}</div>
      {shouldShowBoundingBox && (
        <div
          style={{
            color: boundingBoxColor,
          }}
        >{`Bounding Box: ${boundingBoxDimensionsString}`}</div>
      )}
      {shouldShowScaledBoundingBox && (
        <div
          style={{
            color: boundingBoxColor,
          }}
        >{`Scaled Bounding Box: ${scaledBoundingBoxDimensionsString}`}</div>
      )}
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
