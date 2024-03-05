import { Box } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';

import { InstallModels } from './InstallModels';
import { Model } from './ModelPanel/Model';

export const ModelPane = () => {
  const selectedModelKey = useAppSelector((s) => s.modelmanagerV2.selectedModelKey);
  return (
    <Box layerStyle="first" p={2} borderRadius="base" w="50%" h="full">
      {selectedModelKey ? <Model key={selectedModelKey} /> : <InstallModels />}
    </Box>
  );
};
