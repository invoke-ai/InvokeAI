import { Box, Flex, SimpleGrid } from '@invoke-ai/ui-library';
import { ControlNetOrT2IAdapterDefaultSettings } from 'features/modelManagerV2/subpanels/ModelPanel/ControlNetOrT2IAdapterDefaultSettings/ControlNetOrT2IAdapterDefaultSettings';
import { ModelConvertButton } from 'features/modelManagerV2/subpanels/ModelPanel/ModelConvertButton';
import { ModelEditButton } from 'features/modelManagerV2/subpanels/ModelPanel/ModelEditButton';
import { ModelHeader } from 'features/modelManagerV2/subpanels/ModelPanel/ModelHeader';
import { TriggerPhrases } from 'features/modelManagerV2/subpanels/ModelPanel/TriggerPhrases';
import { memo, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import type { AnyModelConfig } from 'services/api/types';

import { MainModelDefaultSettings } from './MainModelDefaultSettings/MainModelDefaultSettings';
import { ModelAttrView } from './ModelAttrView';

type Props = {
  modelConfig: AnyModelConfig;
};

export const ModelView = memo(({ modelConfig }: Props) => {
  const { t } = useTranslation();
  const withSettings = useMemo(() => {
    if (modelConfig.type === 'main' && modelConfig.base !== 'sdxl-refiner') {
      return true;
    }
    if (modelConfig.type === 'controlnet' || modelConfig.type === 't2i_adapter') {
      return true;
    }
    if (modelConfig.type === 'main' || modelConfig.type === 'lora') {
      return true;
    }

    return false;
  }, [modelConfig.base, modelConfig.type]);

  return (
    <Flex flexDir="column" gap={4}>
      <ModelHeader modelConfig={modelConfig}>
        {modelConfig.format === 'checkpoint' && modelConfig.type === 'main' && (
          <ModelConvertButton modelConfig={modelConfig} />
        )}
        <ModelEditButton />
      </ModelHeader>
      <Flex flexDir="column" h="full" gap={4}>
        <Box layerStyle="second" borderRadius="base" p={4}>
          <SimpleGrid columns={2} gap={4}>
            <ModelAttrView label={t('modelManager.baseModel')} value={modelConfig.base} />
            <ModelAttrView label={t('modelManager.modelType')} value={modelConfig.type} />
            <ModelAttrView label={t('common.format')} value={modelConfig.format} />
            <ModelAttrView label={t('modelManager.path')} value={modelConfig.path} />
            {modelConfig.type === 'main' && (
              <ModelAttrView label={t('modelManager.variant')} value={modelConfig.variant} />
            )}
            {modelConfig.type === 'main' && modelConfig.format === 'diffusers' && modelConfig.repo_variant && (
              <ModelAttrView label={t('modelManager.repoVariant')} value={modelConfig.repo_variant} />
            )}
            {modelConfig.type === 'main' && modelConfig.format === 'checkpoint' && (
              <>
                <ModelAttrView label={t('modelManager.pathToConfig')} value={modelConfig.config_path} />
                <ModelAttrView label={t('modelManager.predictionType')} value={modelConfig.prediction_type} />
                <ModelAttrView label={t('modelManager.upcastAttention')} value={`${modelConfig.upcast_attention}`} />
              </>
            )}
            {modelConfig.type === 'ip_adapter' && modelConfig.format === 'invokeai' && (
              <ModelAttrView label={t('modelManager.imageEncoderModelId')} value={modelConfig.image_encoder_model_id} />
            )}
          </SimpleGrid>
        </Box>
        {withSettings && (
          <Box layerStyle="second" borderRadius="base" p={4}>
            {modelConfig.type === 'main' && modelConfig.base !== 'sdxl-refiner' && (
              <MainModelDefaultSettings modelConfig={modelConfig} />
            )}
            {(modelConfig.type === 'controlnet' || modelConfig.type === 't2i_adapter') && (
              <ControlNetOrT2IAdapterDefaultSettings modelConfig={modelConfig} />
            )}
            {(modelConfig.type === 'main' || modelConfig.type === 'lora') && (
              <TriggerPhrases modelConfig={modelConfig} />
            )}
          </Box>
        )}
      </Flex>
    </Flex>
  );
});

ModelView.displayName = 'ModelView';
