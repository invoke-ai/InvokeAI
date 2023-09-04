import { Box, Flex } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { canvasSelector } from 'features/canvas/store/canvasSelectors';
import GenerationModeStatusText from 'features/parameters/components/Parameters/Canvas/GenerationModeStatusText';
import { isEqual } from 'lodash-es';
import { useTranslation } from 'react-i18next';
import roundToHundreth from '../util/roundToHundreth';
import IAICanvasStatusTextCursorPos from './IAICanvasStatusText/IAICanvasStatusTextCursorPos';
import { memo } from 'react';

const warningColor = 'var(--invokeai-colors-warning-500)';

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
      boundingBoxColor = warningColor;
    }

    const activeLayerColor = layer === 'mask' ? warningColor : 'inherit';

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
    <Flex
      sx={{
        flexDirection: 'column',
        position: 'absolute',
        top: 0,
        insetInlineStart: 0,
        opacity: 0.65,
        display: 'flex',
        fontSize: 'sm',
        padding: 1,
        px: 2,
        minWidth: 48,
        margin: 1,
        borderRadius: 'base',
        pointerEvents: 'none',
        bg: 'base.200',
        _dark: {
          bg: 'base.800',
        },
      }}
    >
      <GenerationModeStatusText />
      <Box
        style={{
          color: activeLayerColor,
        }}
      >{`${t('unifiedCanvas.activeLayer')}: ${activeLayerString}`}</Box>
      <Box>{`${t('unifiedCanvas.canvasScale')}: ${canvasScaleString}%`}</Box>
      {shouldPreserveMaskedArea && (
        <Box
          style={{
            color: warningColor,
          }}
        >
          Preserve Masked Area: On
        </Box>
      )}
      {shouldShowBoundingBox && (
        <Box
          style={{
            color: boundingBoxColor,
          }}
        >{`${t(
          'unifiedCanvas.boundingBox'
        )}: ${boundingBoxDimensionsString}`}</Box>
      )}
      {shouldShowScaledBoundingBox && (
        <Box
          style={{
            color: boundingBoxColor,
          }}
        >{`${t(
          'unifiedCanvas.scaledBoundingBox'
        )}: ${scaledBoundingBoxDimensionsString}`}</Box>
      )}
      {shouldShowCanvasDebugInfo && (
        <>
          <Box>{`${t(
            'unifiedCanvas.boundingBoxPosition'
          )}: ${boundingBoxCoordinatesString}`}</Box>
          <Box>{`${t(
            'unifiedCanvas.canvasDimensions'
          )}: ${canvasDimensionsString}`}</Box>
          <Box>{`${t(
            'unifiedCanvas.canvasPosition'
          )}: ${canvasCoordinatesString}`}</Box>
          <IAICanvasStatusTextCursorPos />
        </>
      )}
    </Flex>
  );
};

export default memo(IAICanvasStatusText);
