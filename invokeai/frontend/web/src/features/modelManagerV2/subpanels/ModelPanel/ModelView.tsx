import { skipToken } from '@reduxjs/toolkit/query';
import { useAppSelector } from '../../../../app/store/storeHooks';
import { useGetModelQuery } from '../../../../services/api/endpoints/models';
import { Flex, Text, Heading } from '@invoke-ai/ui-library';
import DataViewer from '../../../gallery/components/ImageMetadataViewer/DataViewer';
import { useMemo } from 'react';
import {
  CheckpointModelConfig,
  ControlNetConfig,
  DiffusersModelConfig,
  IPAdapterConfig,
  LoRAConfig,
  T2IAdapterConfig,
  TextualInversionConfig,
  VAEConfig,
} from '../../../../services/api/types';
import { ModelAttrView } from './ModelAttrView';

export const ModelView = () => {
  const selectedModelKey = useAppSelector((s) => s.modelmanagerV2.selectedModelKey);
  const { data, isLoading } = useGetModelQuery(selectedModelKey ?? skipToken);

  const modelConfigData = useMemo(() => {
    if (!data) {
      return null;
    }
    const modelFormat = data.config.format;
    const modelType = data.config.type;

    if (modelType === 'main') {
      if (modelFormat === 'diffusers') {
        return data.config as DiffusersModelConfig;
      } else if (modelFormat === 'checkpoint') {
        return data.config as CheckpointModelConfig;
      }
    }

    switch (modelType) {
      case 'lora':
        return data.config as LoRAConfig;
      case 'embedding':
        return data.config as TextualInversionConfig;
      case 't2i_adapter':
        return data.config as T2IAdapterConfig;
      case 'ip_adapter':
        return data.config as IPAdapterConfig;
      case 'controlnet':
        return data.config as ControlNetConfig;
      case 'vae':
        return data.config as VAEConfig;
      default:
        return null;
    }
  }, [data]);

  if (isLoading) {
    return <Text>Loading</Text>;
  }

  if (!modelConfigData) {
    return <Text>Something went wrong</Text>;
  }
  return (
    <Flex flexDir="column" h="full">
      <Flex flexDir="column" gap={1} p={2}>
        <Heading as="h2" fontSize="lg">
          {modelConfigData.name}
        </Heading>
        {modelConfigData.source && <Text variant="subtext">Source: {modelConfigData.source}</Text>}
      </Flex>

      <Flex flexDir="column" p={2} gap={3}>
        <Flex>
          <ModelAttrView label="Description" value={modelConfigData.description} />
        </Flex>
        <Flex gap={2}>
          <ModelAttrView label="Base Model" value={modelConfigData.base} />
          <ModelAttrView label="Model Type" value={modelConfigData.type} />
        </Flex>
        <Flex gap={2}>
          <ModelAttrView label="Format" value={modelConfigData.format} />
          <ModelAttrView label="Path" value={modelConfigData.path} />
        </Flex>
        {modelConfigData.type === 'main' && (
          <>
            <Flex gap={2}>
              {modelConfigData.format === 'diffusers' && (
                <ModelAttrView label="Repo Variant" value={modelConfigData.repo_variant} />
              )}
              {modelConfigData.format === 'checkpoint' && (
                <ModelAttrView label="Config Path" value={modelConfigData.config} />
              )}

              <ModelAttrView label="Variant" value={modelConfigData.variant} />
            </Flex>
            <Flex gap={2}>
              <ModelAttrView label="Prediction Type" value={modelConfigData.prediction_type} />
              <ModelAttrView label="Upcast Attention" value={`${modelConfigData.upcast_attention}`} />
            </Flex>
            <Flex gap={2}>
              <ModelAttrView label="ZTSNR Training" value={`${modelConfigData.ztsnr_training}`} />
              <ModelAttrView label="VAE" value={modelConfigData.vae} />
            </Flex>
          </>
        )}
        {modelConfigData.type === 'ip_adapter' && (
          <Flex gap={2}>
            <ModelAttrView label="Image Encoder Model ID" value={modelConfigData.image_encoder_model_id} />
          </Flex>
        )}
      </Flex>

      <Flex h="full">{!!data?.metadata && <DataViewer label="metadata" data={data.metadata} />}</Flex>
    </Flex>
  );
};
