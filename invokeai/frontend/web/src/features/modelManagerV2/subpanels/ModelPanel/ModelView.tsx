import { Box, Flex, Text } from '@invoke-ai/ui-library';
import { skipToken } from '@reduxjs/toolkit/query';
import { useAppSelector } from 'app/store/storeHooks';
import { TriggerPhrases } from 'features/modelManagerV2/subpanels/ModelPanel/TriggerPhrases';
import { useTranslation } from 'react-i18next';
import { useGetModelConfigQuery } from 'services/api/endpoints/models';

import { DefaultSettings } from './DefaultSettings';
import { ModelAttrView } from './ModelAttrView';

export const ModelView = () => {
  const { t } = useTranslation();
  const selectedModelKey = useAppSelector((s) => s.modelmanagerV2.selectedModelKey);
  const { data, isLoading } = useGetModelConfigQuery(selectedModelKey ?? skipToken);

  if (isLoading) {
    return <Text>{t('common.loading')}</Text>;
  }

  if (!data) {
    return <Text>{t('common.somethingWentWrong')}</Text>;
  }
  return (
    <Flex flexDir="column" h="full" gap={4}>
      <Box layerStyle="second" borderRadius="base" p={4}>
        <Flex flexDir="column" gap={4}>
          <Flex gap={2}>
            <ModelAttrView label={t('modelManager.baseModel')} value={data.base} />
            <ModelAttrView label={t('modelManager.modelType')} value={data.type} />
          </Flex>
          <Flex gap={2}>
            <ModelAttrView label={t('common.format')} value={data.format} />
            <ModelAttrView label={t('modelManager.path')} value={data.path} />
          </Flex>

          {data.type === 'main' && data.format === 'diffusers' && data.repo_variant && (
            <Flex gap={2}>
              <ModelAttrView label={t('modelManager.repoVariant')} value={data.repo_variant} />
            </Flex>
          )}

          {data.type === 'main' && data.format === 'checkpoint' && (
            <>
              <Flex gap={2}>
                <ModelAttrView label={t('modelManager.pathToConfig')} value={data.config_path} />
                <ModelAttrView label={t('modelManager.variant')} value={data.variant} />
              </Flex>
              <Flex gap={2}>
                <ModelAttrView label={t('modelManager.predictionType')} value={data.prediction_type} />
                <ModelAttrView label={t('modelManager.upcastAttention')} value={`${data.upcast_attention}`} />
              </Flex>
            </>
          )}

          {data.type === 'ip_adapter' && (
            <Flex gap={2}>
              <ModelAttrView label={t('modelManager.imageEncoderModelId')} value={data.image_encoder_model_id} />
            </Flex>
          )}
        </Flex>
      </Box>
      {data.type === 'main' && (
        <Box layerStyle="second" borderRadius="base" p={4}>
          <DefaultSettings />
        </Box>
      )}
      {(data.type === 'main' || data.type === 'lora') && (
        <Box layerStyle="second" borderRadius="base" p={4}>
          <TriggerPhrases />
        </Box>
      )}
    </Flex>
  );
};
