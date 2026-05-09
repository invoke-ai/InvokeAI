import type { FormLabelProps } from '@invoke-ai/ui-library';
import { Flex, FormControlGroup } from '@invoke-ai/ui-library';
import { BboxAspectRatioSelect } from 'features/parameters/components/Bbox/BboxAspectRatioSelect';
import { BboxHeight } from 'features/parameters/components/Bbox/BboxHeight';
import { BboxLockAspectRatioButton } from 'features/parameters/components/Bbox/BboxLockAspectRatioButton';
import { BboxPreview } from 'features/parameters/components/Bbox/BboxPreview';
import { BboxSetOptimalSizeButton } from 'features/parameters/components/Bbox/BboxSetOptimalSizeButton';
import { BboxSwapDimensionsButton } from 'features/parameters/components/Bbox/BboxSwapDimensionsButton';
import { BboxWidth } from 'features/parameters/components/Bbox/BboxWidth';
import { memo } from 'react';

export const BboxSettings = memo(() => {
  return (
    <Flex gap={4} alignItems="center">
      <Flex gap={4} flexDirection="column" width="full">
        <FormControlGroup formLabelProps={formLabelProps}>
          <Flex gap={4}>
            <BboxAspectRatioSelect />
            <BboxSwapDimensionsButton />
            <BboxLockAspectRatioButton />
            <BboxSetOptimalSizeButton />
          </Flex>
          <BboxWidth />
          <BboxHeight />
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
