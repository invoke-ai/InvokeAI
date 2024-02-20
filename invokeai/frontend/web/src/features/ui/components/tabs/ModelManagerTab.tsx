import { Box,Flex } from '@invoke-ai/ui-library';
import { ImportModels } from 'features/modelManagerV2/subpanels/ImportModels';
import { ModelManager } from 'features/modelManagerV2/subpanels/ModelManager';
import { memo } from 'react';

const ModelManagerTab = () => {
  return (
    <Box w="full" h="full">
      <Flex w="full" h="full" gap={4}>
        <ModelManager />
        <ImportModels />
      </Flex>
    </Box>
  );
};

export default memo(ModelManagerTab);
