import type { FormLabelProps } from '@invoke-ai/ui-library';
import { Flex, FormControlGroup } from '@invoke-ai/ui-library';
import { memo } from 'react';

import { DimensionsAspectRatioSelect } from './DimensionsAspectRatioSelect';
import { DimensionsHeight } from './DimensionsHeight';
import { DimensionsLockAspectRatioButton } from './DimensionsLockAspectRatioButton';
import { DimensionsPreview } from './DimensionsPreview';
import { DimensionsSetOptimalSizeButton } from './DimensionsSetOptimalSizeButton';
import { DimensionsSwapButton } from './DimensionsSwapButton';
import { DimensionsWidth } from './DimensionsWidth';

export const Dimensions = memo(() => {
  return (
    <Flex gap={4} alignItems="center">
      <Flex gap={4} flexDirection="column" width="full">
        <FormControlGroup formLabelProps={formLabelProps}>
          <Flex gap={4}>
            <DimensionsAspectRatioSelect />
            <DimensionsSwapButton />
            <DimensionsLockAspectRatioButton />
            <DimensionsSetOptimalSizeButton />
          </Flex>
          <DimensionsWidth />
          <DimensionsHeight />
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
