import { Flex } from '@invoke-ai/ui-library';
import { StageComponent } from 'features/regionalPrompts/components/StageComponent';
import { memo } from 'react';

export const AspectRatioPreview = memo(() => {
  return (
    <Flex w="full" h="full" alignItems="center" justifyContent="center" position="relative">
      <StageComponent asPreview />
    </Flex>
  );
});

AspectRatioPreview.displayName = 'AspectRatioPreview';
