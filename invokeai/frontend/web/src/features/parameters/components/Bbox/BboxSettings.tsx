import type { FormLabelProps } from '@invoke-ai/ui-library';
import { Flex, FormControlGroup } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import {
  selectModelSupportsAspectRatio,
  selectModelSupportsPixelDimensions,
} from 'features/controlLayers/store/paramsSlice';
import { BboxAspectRatioSelect } from 'features/parameters/components/Bbox/BboxAspectRatioSelect';
import { BboxHeight } from 'features/parameters/components/Bbox/BboxHeight';
import { BboxLockAspectRatioButton } from 'features/parameters/components/Bbox/BboxLockAspectRatioButton';
import { BboxPreview } from 'features/parameters/components/Bbox/BboxPreview';
import { BboxSetOptimalSizeButton } from 'features/parameters/components/Bbox/BboxSetOptimalSizeButton';
import { BboxSwapDimensionsButton } from 'features/parameters/components/Bbox/BboxSwapDimensionsButton';
import { BboxWidth } from 'features/parameters/components/Bbox/BboxWidth';
import { PixelDimensionsUnsupportedAlert } from 'features/parameters/components/PixelDimensionsUnsupportedAlert';
import { memo } from 'react';

export const BboxSettings = memo(() => {
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
            <BboxAspectRatioSelect />
            <BboxSwapDimensionsButton />
            {supportsPixelDimensions && (
              <>
                <BboxLockAspectRatioButton />
                <BboxSetOptimalSizeButton />
              </>
            )}
          </Flex>
          {supportsPixelDimensions && (
            <>
              <BboxWidth />
              <BboxHeight />
            </>
          )}
          {!supportsPixelDimensions && <PixelDimensionsUnsupportedAlert />}
        </FormControlGroup>
      </Flex>
      <Flex w="108px" h="108px" flexShrink={0} flexGrow={0} alignItems="center" justifyContent="center">
        <BboxPreview />
      </Flex>
    </Flex>
  );
});

BboxSettings.displayName = 'BboxSettings';

const formLabelProps: FormLabelProps = {
  minW: 10,
};
