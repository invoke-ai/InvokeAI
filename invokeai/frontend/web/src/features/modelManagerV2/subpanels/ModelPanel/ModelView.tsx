import { Box, Button, Flex, Text } from '@invoke-ai/ui-library';
import { skipToken } from '@reduxjs/toolkit/query';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { setSelectedModelMode } from 'features/modelManagerV2/store/modelManagerV2Slice';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { IoPencil } from 'react-icons/io5';
import { useGetModelConfigQuery } from 'services/api/endpoints/models';
import type {
  CheckpointModelConfig,
  ControlNetModelConfig,
  DiffusersModelConfig,
  IPAdapterModelConfig,
  LoRAModelConfig,
  T2IAdapterModelConfig,
  TextualInversionModelConfig,
  VAEModelConfig,
} from 'services/api/types';

import { ModelAttrView } from './ModelAttrView';
import { ModelConvert } from './ModelConvert';

export const ModelView = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const selectedModelKey = useAppSelector((s) => s.modelmanagerV2.selectedModelKey);
  const { data, isLoading } = useGetModelConfigQuery(selectedModelKey ?? skipToken);

  const modelData = useMemo(() => {
    if (!data) {
      return null;
    }
    const modelFormat = data.format;
    const modelType = data.type;

    if (modelType === 'main') {
      if (modelFormat === 'diffusers') {
        return data as DiffusersModelConfig;
      } else if (modelFormat === 'checkpoint') {
        return data as CheckpointModelConfig;
      }
    }

    switch (modelType) {
      case 'lora':
        return data as LoRAModelConfig;
      case 'embedding':
        return data as TextualInversionModelConfig;
      case 't2i_adapter':
        return data as T2IAdapterModelConfig;
      case 'ip_adapter':
        return data as IPAdapterModelConfig;
      case 'controlnet':
        return data as ControlNetModelConfig;
      case 'vae':
        return data as VAEModelConfig;
      default:
        return null;
    }
  }, [data]);

  const handleEditModel = useCallback(() => {
    dispatch(setSelectedModelMode('edit'));
  }, [dispatch]);

  if (isLoading) {
    return <Text>{t('common.loading')}</Text>;
  }

  if (!modelData) {
    return <Text>{t('common.somethingWentWrong')}</Text>;
  }
  return (
    <Flex flexDir="column" h="full">
      <Box layerStyle="second" borderRadius="base" p={3}>
        <Flex gap="2" justifyContent="flex-end" w="full">
          <Button size="sm" leftIcon={<IoPencil />} colorScheme="invokeYellow" onClick={handleEditModel}>
            {t('modelManager.edit')}
          </Button>

          {modelData.type === 'main' && modelData.format === 'checkpoint' && <ModelConvert model={modelData} />}
        </Flex>
        <Flex flexDir="column" gap={3}>
          <Flex gap={2}>
            <ModelAttrView label={t('modelManager.baseModel')} value={modelData.base} />
            <ModelAttrView label={t('modelManager.modelType')} value={modelData.type} />
          </Flex>
          <Flex gap={2}>
            <ModelAttrView label={t('common.format')} value={modelData.format} />
            <ModelAttrView label={t('modelManager.path')} value={modelData.path} />
          </Flex>
          {modelData.type === 'main' && (
            <>
              <Flex gap={2}>
                {modelData.format === 'diffusers' && (
                  <ModelAttrView label={t('modelManager.repoVariant')} value={modelData.repo_variant} />
                )}
                {modelData.format === 'checkpoint' && (
                  <ModelAttrView label={t('modelManager.pathToConfig')} value={modelData.config} />
                )}

                <ModelAttrView label={t('modelManager.variant')} value={modelData.variant} />
              </Flex>
              <Flex gap={2}>
                <ModelAttrView label={t('modelManager.predictionType')} value={modelData.prediction_type} />
                <ModelAttrView label={t('modelManager.upcastAttention')} value={`${modelData.upcast_attention}`} />
              </Flex>
              <Flex gap={2}>
                <ModelAttrView label={t('modelManager.ztsnrTraining')} value={`${modelData.ztsnr_training}`} />
                <ModelAttrView label={t('modelManager.vae')} value={modelData.vae} />
              </Flex>
            </>
          )}
          {modelData.type === 'ip_adapter' && (
            <Flex gap={2}>
              <ModelAttrView label={t('modelManager.imageEncoderModelId')} value={modelData.image_encoder_model_id} />
            </Flex>
          )}
        </Flex>
      </Box>
    </Flex>
  );
};
