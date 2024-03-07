import { Flex, Heading, Text } from '@invoke-ai/ui-library';
import { skipToken } from '@reduxjs/toolkit/query';
import { useAppSelector } from 'app/store/storeHooks';
import { ModelConvertButton } from 'features/modelManagerV2/subpanels/ModelPanel/ModelConvertButton';
import { ModelEditButton } from 'features/modelManagerV2/subpanels/ModelPanel/ModelEditButton';
import { useTranslation } from 'react-i18next';
import { useGetModelConfigQuery } from 'services/api/endpoints/models';

import ModelImageUpload from './Fields/ModelImageUpload';
import { ModelEdit } from './ModelEdit';
import { ModelView } from './ModelView';

export const Model = () => {
  const { t } = useTranslation();
  const selectedModelMode = useAppSelector((s) => s.modelmanagerV2.selectedModelMode);
  const selectedModelKey = useAppSelector((s) => s.modelmanagerV2.selectedModelKey);
  const { data, isLoading } = useGetModelConfigQuery(selectedModelKey ?? skipToken);

  if (isLoading) {
    return <Text>{t('common.loading')}</Text>;
  }

  if (!data) {
    return <Text>{t('common.somethingWentWrong')}</Text>;
  }

  return (
    <Flex flexDir="column" gap={4}>
      <Flex alignItems="flex-start" gap={4}>
        <ModelImageUpload model_key={selectedModelKey} model_image={data.cover_image} />
        <Flex flexDir="column" gap={1} flexGrow={1}>
          <Flex gap={2} position="relative">
            <Heading as="h2" fontSize="lg" w="full">
              {data.name}
            </Heading>
            <Flex position="absolute" gap={2} right={0} top={0}>
              <ModelEditButton />
              <ModelConvertButton modelKey={selectedModelKey} />
            </Flex>
          </Flex>
          {data.source && (
            <Text variant="subtext">
              {t('modelManager.source')}: {data?.source}
            </Text>
          )}
          <Text noOfLines={3}>{data.description}</Text>
        </Flex>
      </Flex>
      {selectedModelMode === 'view' ? <ModelView /> : <ModelEdit />}
    </Flex>
  );
};
