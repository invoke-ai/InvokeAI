import { Box } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { selectSelectedModelKey } from 'features/modelManagerV2/store/modelManagerV2Slice';
import { memo } from 'react';

import { InstallModels } from './InstallModels';
import { Model } from './ModelPanel/Model';

export const ModelPane = memo(() => {
  const selectedModelKey = useAppSelector(selectSelectedModelKey);
  return (
    <Box layerStyle="first" p={4} borderRadius="base" w="50%" h="full">
      {selectedModelKey ? <Model key={selectedModelKey} /> : <InstallModels />}
    </Box>
  );
});

ModelPane.displayName = 'ModelPane';
