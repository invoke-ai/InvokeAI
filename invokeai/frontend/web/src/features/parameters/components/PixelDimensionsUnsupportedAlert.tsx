import type { FormLabelProps } from '@invoke-ai/ui-library';
import { Alert, Flex, FormControlGroup, Text } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import {
  selectModelSupportsAspectRatio,
  selectModelSupportsPixelDimensions,
} from 'features/controlLayers/store/paramsSlice';
import { memo } from 'react';

import { DimensionsAspectRatioSelect } from './DimensionsAspectRatioSelect';
import { DimensionsHeight } from './DimensionsHeight';
import { DimensionsLockAspectRatioButton } from './DimensionsLockAspectRatioButton';
import { DimensionsPreview } from './DimensionsPreview';
import { DimensionsSetOptimalSizeButton } from './DimensionsSetOptimalSizeButton';
import { DimensionsSwapButton } from './DimensionsSwapButton';
import { DimensionsWidth } from './DimensionsWidth';

export const PixelDimensionsUnsupportedAlert = memo(() => {
  return (
    <Alert status="info" borderRadius="base" flexDir="column" gap={2} overflow="unset">
      <Text fontSize="md">This model does not support user-defined width and height.</Text>
    </Alert>
  );
});

PixelDimensionsUnsupportedAlert.displayName = 'PixelDimensionsUnsupportedAlert';
