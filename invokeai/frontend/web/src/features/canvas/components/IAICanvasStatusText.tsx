import { Box, Flex } from '@invoke-ai/ui-library';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { selectCanvasSlice } from 'features/canvas/store/canvasSlice';
import roundToHundreth from 'features/canvas/util/roundToHundreth';
import GenerationModeStatusText from 'features/parameters/components/Canvas/GenerationModeStatusText';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

import IAICanvasStatusTextCursorPos from './IAICanvasStatusText/IAICanvasStatusTextCursorPos';

const warningColor = 'var(--invoke-colors-warning-500)';

const selector = createMemoizedSelector(selectCanvasSlice, (canvas) => {
  const {
    stageDimensions: { width: stageWidth, height: stageHeight },
    stageCoordinates: { x: stageX, y: stageY },
    boundingBoxDimensions: { width: boxWidth, height: boxHeight },
    scaledBoundingBoxDimensions: { width: scaledBoxWidth, height: scaledBoxHeight },
    boundingBoxCoordinates: { x: boxX, y: boxY },
    stageScale,
    shouldShowCanvasDebugInfo,
    layer,
    boundingBoxScaleMethod,
    shouldPreserveMaskedArea,
  } = canvas;

  let boundingBoxColor = 'inherit';

  if (
    (boundingBoxScaleMethod === 'none' && (boxWidth < 512 || boxHeight < 512)) ||
    (boundingBoxScaleMethod === 'manual' && scaledBoxWidth * scaledBoxHeight < 512 * 512)
  ) {
    boundingBoxColor = warningColor;
  }

  const activeLayerColor = layer === 'mask' ? warningColor : 'inherit';

  return {
    activeLayerColor,
    layer,
    boundingBoxColor,
    boundingBoxCoordinatesString: `(${roundToHundreth(boxX)}, ${roundToHundreth(boxY)})`,
    boundingBoxDimensionsString: `${boxWidth}×${boxHeight}`,
    scaledBoundingBoxDimensionsString: `${scaledBoxWidth}×${scaledBoxHeight}`,
    canvasCoordinatesString: `${roundToHundreth(stageX)}×${roundToHundreth(stageY)}`,
    canvasDimensionsString: `${stageWidth}×${stageHeight}`,
    canvasScaleString: Math.round(stageScale * 100),
    shouldShowCanvasDebugInfo,
    shouldShowBoundingBox: boundingBoxScaleMethod !== 'auto',
    shouldShowScaledBoundingBox: boundingBoxScaleMethod !== 'none',
    shouldPreserveMaskedArea,
  };
});

const IAICanvasStatusText = () => {
  const {
    activeLayerColor,
    layer,
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
      flexDirection="column"
      position="absolute"
      top={0}
      insetInlineStart={0}
      opacity={0.65}
      display="flex"
      fontSize="sm"
      padding={1}
      px={2}
      minWidth={48}
      margin={1}
      borderRadius="base"
      pointerEvents="none"
      bg="base.800"
    >
      <GenerationModeStatusText />
      <Box color={activeLayerColor}>{`${t('unifiedCanvas.activeLayer')}: ${t(`unifiedCanvas.${layer}`)}`}</Box>
      <Box>{`${t('unifiedCanvas.canvasScale')}: ${canvasScaleString}%`}</Box>
      {shouldPreserveMaskedArea && (
        <Box color={warningColor}>
          {t('unifiedCanvas.preserveMaskedArea')}: {t('common.on')}
        </Box>
      )}
      {shouldShowBoundingBox && (
        <Box color={boundingBoxColor}>{`${t('unifiedCanvas.boundingBox')}: ${boundingBoxDimensionsString}`}</Box>
      )}
      {shouldShowScaledBoundingBox && (
        <Box color={boundingBoxColor}>{`${t(
          'unifiedCanvas.scaledBoundingBox'
        )}: ${scaledBoundingBoxDimensionsString}`}</Box>
      )}
      {shouldShowCanvasDebugInfo && (
        <>
          <Box>{`${t('unifiedCanvas.boundingBoxPosition')}: ${boundingBoxCoordinatesString}`}</Box>
          <Box>{`${t('unifiedCanvas.canvasDimensions')}: ${canvasDimensionsString}`}</Box>
          <Box>{`${t('unifiedCanvas.canvasPosition')}: ${canvasCoordinatesString}`}</Box>
          <IAICanvasStatusTextCursorPos />
        </>
      )}
    </Flex>
  );
};

export default memo(IAICanvasStatusText);
