import { Flex } from '@chakra-ui/react';
import { memo } from 'react';

import FoundModelsList from './FoundModelsList';
import ScanAdvancedAddModels from './ScanAdvancedAddModels';
import SearchFolderForm from './SearchFolderForm';

const ScanModels = () => {
  return (
    <Flex flexDirection="column" w="100%" gap={4}>
      <SearchFolderForm />
      <Flex gap={4}>
        <Flex
          maxHeight="calc(100vh - 300px)"
          overflow="scroll"
          gap={4}
          w="100%"
        >
          <FoundModelsList />
        </Flex>
        <ScanAdvancedAddModels />
      </Flex>
    </Flex>
  );
};

export default memo(ScanModels);
