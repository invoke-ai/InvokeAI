import { Box, Divider, Flex, SimpleGrid } from '@invoke-ai/ui-library';
import { useIsModelManagerEnabled } from 'features/modelManagerV2/hooks/useIsModelManagerEnabled';
import { ControlAdapterModelDefaultSettings } from 'features/modelManagerV2/subpanels/ModelPanel/ControlAdapterModelDefaultSettings/ControlAdapterModelDefaultSettings';
import { EncoderModelSettings } from 'features/modelManagerV2/subpanels/ModelPanel/EncoderModelSettings/EncoderModelSettings';
import { LoRAModelDefaultSettings } from 'features/modelManagerV2/subpanels/ModelPanel/LoRAModelDefaultSettings/LoRAModelDefaultSettings';
import { ModelConvertButton } from 'features/modelManagerV2/subpanels/ModelPanel/ModelConvertButton';
import { ModelEditButton } from 'features/modelManagerV2/subpanels/ModelPanel/ModelEditButton';
import { ModelHeader } from 'features/modelManagerV2/subpanels/ModelPanel/ModelHeader';
import { TriggerPhrases } from 'features/modelManagerV2/subpanels/ModelPanel/TriggerPhrases';
import { filesize } from 'filesize';
import { memo, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import type {
  AnyModelConfig,
  CLIPEmbedModelConfig,
  CLIPVisionModelConfig,
  LlavaOnevisionModelConfig,
  Qwen3EncoderModelConfig,
  SigLIPModelConfig,
  T5EncoderModelConfig,
} from 'services/api/types';

import { isExternalModel } from './isExternalModel';
import { MainModelDefaultSettings } from './MainModelDefaultSettings/MainModelDefaultSettings';
import { ModelAttrView } from './ModelAttrView';
import { ModelDeleteButton } from './ModelDeleteButton';
import { ModelReidentifyButton } from './ModelReidentifyButton';
import { ModelUpdatePathButton } from './ModelUpdatePathButton';
import { RelatedModels } from './RelatedModels';

type EncoderModelConfig =
  | CLIPEmbedModelConfig
  | T5EncoderModelConfig
  | Qwen3EncoderModelConfig
  | CLIPVisionModelConfig
  | SigLIPModelConfig
  | LlavaOnevisionModelConfig;

const isEncoderModel = (modelConfig: AnyModelConfig): modelConfig is EncoderModelConfig => {
  return (
    modelConfig.type === 'clip_embed' ||
    modelConfig.type === 't5_encoder' ||
    modelConfig.type === 'qwen3_encoder' ||
    modelConfig.type === 'clip_vision' ||
    modelConfig.type === 'siglip' ||
    modelConfig.type === 'llava_onevision'
  );
};

type Props = {
  modelConfig: AnyModelConfig;
};

export const ModelView = memo(({ modelConfig }: Props) => {
  const { t } = useTranslation();
  const canManageModels = useIsModelManagerEnabled();

  // Only allow path updates for external models (not Invoke-controlled)
  const canUpdatePath = useMemo(() => isExternalModel(modelConfig.path), [modelConfig.path]);

  const withSettings = useMemo(() => {
    if (modelConfig.type === 'main' && modelConfig.base !== 'sdxl-refiner') {
      return true;
    }
    if (
      modelConfig.type === 'controlnet' ||
      modelConfig.type === 't2i_adapter' ||
      modelConfig.type === 'control_lora'
    ) {
      return true;
    }
    if (modelConfig.type === 'main' || modelConfig.type === 'lora') {
      return true;
    }
    // Encoder models
    if (isEncoderModel(modelConfig)) {
      return true;
    }

    return false;
  }, [modelConfig]);

  return (
    <Flex flexDir="column" gap={4} h="full">
      <ModelHeader modelConfig={modelConfig}>
        {canManageModels && canUpdatePath && <ModelUpdatePathButton modelConfig={modelConfig} />}
        {canManageModels && <ModelReidentifyButton modelConfig={modelConfig} />}
        {canManageModels && modelConfig.format === 'checkpoint' && modelConfig.type === 'main' && (
          <ModelConvertButton modelConfig={modelConfig} />
        )}
        {canManageModels && <ModelEditButton />}
        {canManageModels && <ModelDeleteButton modelConfig={modelConfig} />}
      </ModelHeader>
      <Divider />
      <Flex flexDir="column" gap={4}>
        <Box>
          <SimpleGrid columns={2} gap={4}>
            <ModelAttrView label={t('modelManager.baseModel')} value={modelConfig.base} />
            <ModelAttrView label={t('modelManager.modelType')} value={modelConfig.type} />
            <ModelAttrView label={t('modelManager.modelFormat')} value={modelConfig.format} />
            <ModelAttrView label={t('modelManager.path')} value={modelConfig.path} />
            <ModelAttrView label={t('modelManager.fileSize')} value={filesize(modelConfig.file_size)} />
            {'variant' in modelConfig && modelConfig.variant && (
              <ModelAttrView label={t('modelManager.variant')} value={modelConfig.variant} />
            )}
            {modelConfig.type === 'main' && modelConfig.format === 'diffusers' && modelConfig.repo_variant && (
              <ModelAttrView label={t('modelManager.repoVariant')} value={modelConfig.repo_variant} />
            )}
            {modelConfig.type === 'main' && modelConfig.format === 'checkpoint' && (
              <ModelAttrView label={t('modelManager.pathToConfig')} value={modelConfig.config_path} />
            )}
            {modelConfig.type === 'main' && modelConfig.format === 'checkpoint' && 'prediction_type' in modelConfig && (
              <ModelAttrView label={t('modelManager.predictionType')} value={modelConfig.prediction_type} />
            )}
            {modelConfig.type === 'ip_adapter' && modelConfig.format === 'invokeai' && (
              <ModelAttrView label={t('modelManager.imageEncoderModelId')} value={modelConfig.image_encoder_model_id} />
            )}
          </SimpleGrid>
        </Box>
        {withSettings && (
          <>
            <Divider />
            <Box>
              {modelConfig.type === 'main' && modelConfig.base !== 'sdxl-refiner' && (
                <MainModelDefaultSettings modelConfig={modelConfig} />
              )}
              {(modelConfig.type === 'controlnet' ||
                modelConfig.type === 't2i_adapter' ||
                modelConfig.type === 'control_lora') && (
                <ControlAdapterModelDefaultSettings modelConfig={modelConfig} />
              )}
              {modelConfig.type === 'lora' && (
                <>
                  <LoRAModelDefaultSettings modelConfig={modelConfig} />
                  <TriggerPhrases modelConfig={modelConfig} />
                </>
              )}
              {modelConfig.type === 'main' && <TriggerPhrases modelConfig={modelConfig} />}
              {isEncoderModel(modelConfig) && <EncoderModelSettings modelConfig={modelConfig} />}
            </Box>
          </>
        )}
        <Divider />
        <Box overflowY="auto">
          <RelatedModels modelConfig={modelConfig} />
        </Box>
      </Flex>
    </Flex>
  );
});

ModelView.displayName = 'ModelView';
