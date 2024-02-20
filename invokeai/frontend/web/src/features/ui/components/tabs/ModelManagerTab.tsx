import { Flex, Box } from '@invoke-ai/ui-library';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { ImportModels } from '../../../modelManagerV2/subpanels/ImportModels';
import { ModelManager } from '../../../modelManagerV2/subpanels/ModelManager';

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
