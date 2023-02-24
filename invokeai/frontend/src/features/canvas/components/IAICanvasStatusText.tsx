import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/storeHooks';
import { canvasSelector } from 'features/canvas/store/canvasSelectors';
import { isEqual } from 'lodash';

import { useTranslation } from 'react-i18next';
import roundToHundreth from '../util/roundToHundreth';
import IAICanvasStatusTextCursorPos from './IAICanvasStatusText/IAICanvasStatusTextCursorPos';

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
      shouldPreserveMaskedArea,
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
      shouldPreserveMaskedArea,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: isEqual,
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
    shouldPreserveMaskedArea,
  } = useAppSelector(selector);

  const { t } = useTranslation();

  return (
    <div className="canvas-status-text">
      <div
        style={{
          color: activeLayerColor,
        }}
      >{`${t('unifiedCanvas.activeLayer')}: ${activeLayerString}`}</div>
      <div>{`${t('unifiedCanvas.canvasScale')}: ${canvasScaleString}%`}</div>
      {shouldPreserveMaskedArea && (
        <div
          style={{
            color: 'var(--status-working-color)',
          }}
        >
          Preserve Masked Area: On
        </div>
      )}
      {shouldShowBoundingBox && (
        <div
          style={{
            color: boundingBoxColor,
          }}
        >{`${t(
          'unifiedCanvas.boundingBox'
        )}: ${boundingBoxDimensionsString}`}</div>
      )}
      {shouldShowScaledBoundingBox && (
        <div
          style={{
            color: boundingBoxColor,
          }}
        >{`${t(
          'unifiedCanvas.scaledBoundingBox'
        )}: ${scaledBoundingBoxDimensionsString}`}</div>
      )}
      {shouldShowCanvasDebugInfo && (
        <>
          <div>{`${t(
            'unifiedCanvas.boundingBoxPosition'
          )}: ${boundingBoxCoordinatesString}`}</div>
          <div>{`${t(
            'unifiedCanvas.canvasDimensions'
          )}: ${canvasDimensionsString}`}</div>
          <div>{`${t(
            'unifiedCanvas.canvasPosition'
          )}: ${canvasCoordinatesString}`}</div>
          <IAICanvasStatusTextCursorPos />
        </>
      )}
    </div>
  );
};

export default IAICanvasStatusText;
