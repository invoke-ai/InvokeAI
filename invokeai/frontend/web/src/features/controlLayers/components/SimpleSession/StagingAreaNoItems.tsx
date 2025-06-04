/* eslint-disable i18next/no-literal-string */

import { Flex, Text } from '@invoke-ai/ui-library';
import { memo } from 'react';

export const StagingAreaNoItems = memo(() => {
  return (
    <Flex w="full" h="full" alignItems="center" justifyContent="center">
      <Text>No generations</Text>
    </Flex>
  );
});
StagingAreaNoItems.displayName = 'StagingAreaNoItems';
