import { Flex } from '@invoke-ai/ui-library';
import { memo } from 'react';

import SyncModels from './ModelManagerSettingsPanel/SyncModels';

const ModelManagerSettingsPanel = () => {
  return (
    <Flex>
      <SyncModels />
    </Flex>
  );
};

export default memo(ModelManagerSettingsPanel);
