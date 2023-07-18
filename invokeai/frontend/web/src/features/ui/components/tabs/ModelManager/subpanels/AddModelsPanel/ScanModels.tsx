import { Flex } from '@chakra-ui/react';
import FoundModelsList from './FoundModelsList';
import ScanAdvancedAddModels from './ScanAdvancedAddModels';
import SearchFolderForm from './SearchFolderForm';

export default function ScanModels() {
  return (
    <Flex flexDirection="column" w="100%" gap={4}>
      <SearchFolderForm />
      <Flex gap={4}>
        <Flex
          sx={{
            maxHeight: window.innerHeight - 300,
            overflow: 'scroll',
            gap: 4,
            w: '100%',
          }}
        >
          <FoundModelsList />
        </Flex>
        <ScanAdvancedAddModels />
      </Flex>
    </Flex>
  );
}
