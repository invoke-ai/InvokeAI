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
import { PixelDimensionsUnsupportedAlert } from '../PixelDimensionsUnsupportedAlert';

export const Dimensions = memo(() => {
  const supportsAspectRatio = useAppSelector(selectModelSupportsAspectRatio);
  const supportsPixelDimensions = useAppSelector(selectModelSupportsPixelDimensions);

  if (!supportsAspectRatio) {
    return null;
  }

  return (
    <Flex gap={4} alignItems="center">
      <Flex gap={4} flexDirection="column" width="full">
        <FormControlGroup formLabelProps={formLabelProps}>
          <Flex gap={4}>
            <DimensionsAspectRatioSelect />
            <DimensionsSwapButton />
            {supportsPixelDimensions && (
              <>
                <DimensionsLockAspectRatioButton />
                <DimensionsSetOptimalSizeButton />
              </>
            )}
          </Flex>
          {supportsPixelDimensions && (
            <>
              <DimensionsWidth />
              <DimensionsHeight />
            </>
          )}
          {!supportsPixelDimensions && <PixelDimensionsUnsupportedAlert />}
        </FormControlGroup>
      </Flex>
      <Flex w="108px" h="108px" flexShrink={0} flexGrow={0} alignItems="center" justifyContent="center">
        <DimensionsPreview />
      </Flex>
    </Flex>
  );
});

Dimensions.displayName = 'Dimensions';

const formLabelProps: FormLabelProps = {
  minW: 10,
};
