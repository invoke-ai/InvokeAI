import { Flex } from '@chakra-ui/react';
import { ensureModelsLoaded } from '@features/models/data/modelsStore';
import { ModelInstallRuntime } from '@features/models/ui/ModelInstallRuntime';
import { useMountEffect } from '@platform/react/useMountEffect';

import { DetailPane } from './manager/DetailPane';
import { LibraryColumn } from './manager/LibraryColumn';

/** Full model manager: persistent library column plus right-side detail pane. */
export const ModelManagerView = () => {
  useMountEffect(() => {
    void ensureModelsLoaded();
  });

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
