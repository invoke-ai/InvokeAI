import { Flex } from '@invoke-ai/ui-library';
import { memo } from 'react';

import FoundModelsList from './FoundModelsList';
import ScanAdvancedAddModels from './ScanAdvancedAddModels';
import SearchFolderForm from './SearchFolderForm';

const ScanModels = () => {
  return (
    <Flex flexDirection="column" w="100%" h="full" gap={4}>
      <SearchFolderForm />
      <Flex gap={4}>
        <Flex overflow="scroll" gap={4} w="100%" h="full">
          <FoundModelsList />
        </Flex>
        <ScanAdvancedAddModels />
      </Flex>
    </Flex>
  );
};

export default memo(ScanModels);
