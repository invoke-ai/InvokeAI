import { Flex } from '@chakra-ui/react';
import FoundModelsList from './FoundModelsList';
import SearchFolderForm from './SearchFolderForm';

export default function ScanModels() {
  return (
    <Flex flexDirection="column" w="100%" gap={2}>
      <SearchFolderForm />
      <Flex sx={{ maxHeight: window.innerHeight - 330, overflow: 'scroll' }}>
        <FoundModelsList />
      </Flex>
    </Flex>
  );
}
