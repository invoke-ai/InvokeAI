/* eslint-disable i18next/no-literal-string */
import { Flex, Heading, Spacer } from '@invoke-ai/ui-library';
import { StartOverButton } from 'features/controlLayers/components/StartOverButton';
import { memo } from 'react';

export const StagingAreaHeader = memo(() => {
  return (
    <Flex gap={2} w="full" alignItems="center" px={2}>
      <Heading size="sm">Review Session</Heading>
      <Spacer />
      <StartOverButton />
    </Flex>
  );
});
StagingAreaHeader.displayName = 'StagingAreaHeader';
