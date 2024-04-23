import { Flex } from '@invoke-ai/ui-library';
import { StageComponent } from 'features/regionalPrompts/components/StageComponent';

export const AspectRatioPreview = () => {
  return (
    <Flex w="full" h="full" alignItems="center" justifyContent="center">
      <StageComponent asPreview />
    </Flex>
  );
};
