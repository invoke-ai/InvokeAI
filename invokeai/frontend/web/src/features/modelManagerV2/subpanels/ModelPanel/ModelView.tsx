import { Box, Flex, SimpleGrid, Text } from '@invoke-ai/ui-library';
import { skipToken } from '@reduxjs/toolkit/query';
import { useAppSelector } from 'app/store/storeHooks';
import { ControlNetOrT2IAdapterDefaultSettings } from 'features/modelManagerV2/subpanels/ModelPanel/ControlNetOrT2IAdapterDefaultSettings/ControlNetOrT2IAdapterDefaultSettings';
import { TriggerPhrases } from 'features/modelManagerV2/subpanels/ModelPanel/TriggerPhrases';
import { useTranslation } from 'react-i18next';
import { useGetModelConfigQuery } from 'services/api/endpoints/models';

import { MainModelDefaultSettings } from './MainModelDefaultSettings/MainModelDefaultSettings';
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
        <SimpleGrid columns={2} gap={4}>
          <ModelAttrView label={t('modelManager.baseModel')} value={data.base} />
          <ModelAttrView label={t('modelManager.modelType')} value={data.type} />
          <ModelAttrView label={t('common.format')} value={data.format} />
          <ModelAttrView label={t('modelManager.path')} value={data.path} />
          {data.type === 'main' && <ModelAttrView label={t('modelManager.variant')} value={data.variant} />}
          {data.type === 'main' && data.format === 'diffusers' && data.repo_variant && (
            <ModelAttrView label={t('modelManager.repoVariant')} value={data.repo_variant} />
          )}
          {data.type === 'main' && data.format === 'checkpoint' && (
            <>
              <ModelAttrView label={t('modelManager.pathToConfig')} value={data.config_path} />
              <ModelAttrView label={t('modelManager.predictionType')} value={data.prediction_type} />
              <ModelAttrView label={t('modelManager.upcastAttention')} value={`${data.upcast_attention}`} />
            </>
          )}
          {data.type === 'ip_adapter' && data.format === 'invokeai' && (
            <ModelAttrView label={t('modelManager.imageEncoderModelId')} value={data.image_encoder_model_id} />
          )}
        </SimpleGrid>
      </Box>
      <Box layerStyle="second" borderRadius="base" p={4}>
        {data.type === 'main' && data.base !== 'sdxl-refiner' && <MainModelDefaultSettings />}
        {(data.type === 'controlnet' || data.type === 't2i_adapter') && <ControlNetOrT2IAdapterDefaultSettings />}
        {(data.type === 'main' || data.type === 'lora') && <TriggerPhrases />}
      </Box>
    </Flex>
  );
};
