/* eslint-disable i18next/no-literal-string */
import { Divider, Flex } from '@invoke-ai/ui-library';
import { StagingAreaItemsList } from 'features/controlLayers/components/SimpleSession/StagingAreaItemsList';
import { StagingAreaSelectedItem } from 'features/controlLayers/components/SimpleSession/StagingAreaSelectedItem';
import { memo } from 'react';

export const StagingAreaContent = memo(() => {
  return (
    <>
      <Flex position="relative" w="full" h="full" maxH="full" alignItems="center" justifyContent="center" minH={0}>
        <StagingAreaSelectedItem />
      </Flex>
      <Divider />
      <Flex position="relative" maxW="full" w="full" h={108}>
        <StagingAreaItemsList />
      </Flex>
    </>
  );
});
StagingAreaContent.displayName = 'StagingAreaContent';
