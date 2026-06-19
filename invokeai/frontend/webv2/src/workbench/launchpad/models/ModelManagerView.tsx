import { Flex } from '@chakra-ui/react';
import { ModelInstallRuntime } from '@workbench/models/ModelInstallRuntime';
import { ensureModelsLoaded } from '@workbench/models/modelsStore';
import { useEffect } from 'react';

import { DetailPane } from './manager/DetailPane';
import { LibraryColumn } from './manager/LibraryColumn';

/** Full model manager: persistent library column plus right-side detail pane. */
export const ModelManagerView = () => {
  useEffect(() => {
    ensureModelsLoaded();
  }, []);

  return (
    <Flex direction="column" h="full" minH="0" w="full">
      <ModelInstallRuntime />
      <Flex flex="1" minH="0" w="full">
        <LibraryColumn />
        <DetailPane />
      </Flex>
    </Flex>
  );
};
