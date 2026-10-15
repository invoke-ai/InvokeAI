import { Flex } from '@invoke-ai/ui-library';
import { FetchingModelsLoader } from 'features/modelManagerV2/subpanels/ModelManagerPanel/FetchingModelsLoader';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { useGetStarterModelsQuery } from 'services/api/endpoints/models';

import { StarterModelsResults } from './StarterModelsResults';

export const StarterModelsForm = memo(() => {
  const { isLoading, data } = useGetStarterModelsQuery();
  const { t } = useTranslation();

  return (
    <Flex flexDir="column" height="100%" gap={3}>
      {isLoading && <FetchingModelsLoader loadingMessage={t('common.loading')} />}
      {data && <StarterModelsResults results={data} />}
    </Flex>
  );
});

StarterModelsForm.displayName = 'StarterModelsForm';
