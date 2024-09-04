import { Flex } from '@invoke-ai/ui-library';
import { EMPTY_ARRAY } from 'app/store/constants';
import { FetchingModelsLoader } from 'features/modelManagerV2/subpanels/ModelManagerPanel/FetchingModelsLoader';
import { memo, useMemo } from 'react';
import {
  modelConfigsAdapterSelectors,
  useGetModelConfigsQuery,
  useGetStarterModelsQuery,
} from 'services/api/endpoints/models';

import { StarterModelsResults } from './StarterModelsResults';

export const StarterModelsForm = memo(() => {
  const { isLoading, data } = useGetStarterModelsQuery();
  const { data: modelListRes } = useGetModelConfigsQuery();

  const modelList = useMemo(() => {
    if (!modelListRes) {
      return EMPTY_ARRAY;
    }

    return modelConfigsAdapterSelectors.selectAll(modelListRes);
  }, [modelListRes]);

  return (
    <Flex flexDir="column" height="100%" gap={3}>
      {isLoading && <FetchingModelsLoader loadingMessage="Loading Embeddings..." />}
      {data && <StarterModelsResults results={data} modelList={modelList} />}
    </Flex>
  );
});

StarterModelsForm.displayName = 'StarterModelsForm';
