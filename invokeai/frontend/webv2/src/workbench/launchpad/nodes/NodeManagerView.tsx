import { Flex } from '@chakra-ui/react';
import { ensureCustomNodePacksLoaded } from '@workbench/customNodes/nodesStore';
import { useEffect } from 'react';

import { DetailPane } from './manager/DetailPane';
import { LibraryColumn } from './manager/LibraryColumn';

/** Full custom nodes manager: persistent pack library plus right-side detail pane. */
export const NodeManagerView = () => {
  useEffect(() => {
    ensureCustomNodePacksLoaded();
  }, []);

  return (
    <Flex direction="column" h="full" minH="0" w="full">
      <Flex flex="1" minH="0" w="full">
        <LibraryColumn />
        <DetailPane />
      </Flex>
    </Flex>
  );
};
