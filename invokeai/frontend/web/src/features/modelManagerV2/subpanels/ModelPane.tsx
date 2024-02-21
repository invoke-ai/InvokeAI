import { Box } from '@invoke-ai/ui-library';
import { useAppSelector } from '../../../app/store/storeHooks';
import { ImportModels } from './ImportModels';
import { ModelView } from './ModelPanel/ModelView';

export const ModelPane = () => {
  const selectedModelKey = useAppSelector((s) => s.modelmanagerV2.selectedModelKey);
  return (
    <Box layerStyle="first" p={2} borderRadius="base" w="full" h="full">
      {selectedModelKey ? <ModelView /> : <ImportModels />}
    </Box>
  );
};
