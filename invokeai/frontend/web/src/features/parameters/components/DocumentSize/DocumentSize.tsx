import type { FormLabelProps } from '@invoke-ai/ui-library';
import { Flex, FormControlGroup } from '@invoke-ai/ui-library';
import { ParamHeight } from 'features/parameters/components/Core/ParamHeight';
import { ParamWidth } from 'features/parameters/components/Core/ParamWidth';
import { AspectRatioIconPreview } from 'features/parameters/components/DocumentSize/AspectRatioIconPreview';
import { AspectRatioSelect } from 'features/parameters/components/DocumentSize/AspectRatioSelect';
import { LockAspectRatioButton } from 'features/parameters/components/DocumentSize/LockAspectRatioButton';
import { SetOptimalSizeButton } from 'features/parameters/components/DocumentSize/SetOptimalSizeButton';
import { SwapDimensionsButton } from 'features/parameters/components/DocumentSize/SwapDimensionsButton';
import { memo } from 'react';

export const DocumentSize = memo(() => {
  return (
    <Flex gap={4} alignItems="center">
      <Flex gap={4} flexDirection="column" width="full">
        <FormControlGroup formLabelProps={formLabelProps}>
          <Flex gap={4}>
            <AspectRatioSelect />
            <SwapDimensionsButton />
            <LockAspectRatioButton />
            <SetOptimalSizeButton />
          </Flex>
          <ParamWidth />
          <ParamHeight />
        </FormControlGroup>
      </Flex>
      <Flex w="108px" h="108px" flexShrink={0} flexGrow={0}>
        <AspectRatioIconPreview />
      </Flex>
    </Flex>
  );
});

DocumentSize.displayName = 'DocumentSize';

const formLabelProps: FormLabelProps = {
  minW: 14,
};
