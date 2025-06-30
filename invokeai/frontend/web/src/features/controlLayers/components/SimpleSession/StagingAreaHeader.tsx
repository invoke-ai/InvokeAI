import { Flex, Heading, Spacer } from '@invoke-ai/ui-library';
import { memo } from 'react';

export const StagingAreaHeader = memo(() => {
  return (
    <Flex gap={2} w="full" alignItems="center" px={2}>
      <Heading size="sm">Review Session</Heading>
      <Spacer />
    </Flex>
  );
});
StagingAreaHeader.displayName = 'StagingAreaHeader';
