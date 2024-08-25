import { Box } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { memo } from 'react';

import { InstallModels } from './InstallModels';
import { Model } from './ModelPanel/Model';

export const ModelPane = memo(() => {
  const selectedModelKey = useAppSelector((s) => s.modelmanagerV2.selectedModelKey);
  return (
    <Box layerStyle="body" pt={4} borderRadius="base" w="50%" h="full">
      {selectedModelKey ? <Model key={selectedModelKey} /> : <InstallModels />}
    </Box>
  );
});

ModelPane.displayName = 'ModelPane';
