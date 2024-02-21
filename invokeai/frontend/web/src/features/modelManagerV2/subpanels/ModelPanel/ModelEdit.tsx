import { skipToken } from '@reduxjs/toolkit/query';
import { useAppDispatch, useAppSelector } from '../../../../app/store/storeHooks';
import { useGetModelQuery } from '../../../../services/api/endpoints/models';
import { Flex, Text, Heading, Button, Input, FormControl, FormLabel, Textarea } from '@invoke-ai/ui-library';
import { useCallback, useMemo } from 'react';
import {
  AnyModelConfig,
  CheckpointModelConfig,
  ControlNetConfig,
  DiffusersModelConfig,
  IPAdapterConfig,
  LoRAConfig,
  T2IAdapterConfig,
  TextualInversionConfig,
  VAEConfig,
} from '../../../../services/api/types';
import { setSelectedModelMode } from '../../store/modelManagerV2Slice';
import BaseModelSelect from './Fields/BaseModelSelect';
import { useForm } from 'react-hook-form';
import ModelTypeSelect from './Fields/ModelTypeSelect';
import ModelVariantSelect from './Fields/ModelVariantSelect';
import RepoVariantSelect from './Fields/RepoVariantSelect';
import PredictionTypeSelect from './Fields/PredictionTypeSelect';
import BooleanSelect from './Fields/BooleanSelect';
import ModelFormatSelect from './Fields/ModelFormatSelect';

export const ModelEdit = () => {
  const dispatch = useAppDispatch();
  const selectedModelKey = useAppSelector((s) => s.modelmanagerV2.selectedModelKey);
  const { data, isLoading } = useGetModelQuery(selectedModelKey ?? skipToken);

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
        return data as LoRAConfig;
      case 'embedding':
        return data as TextualInversionConfig;
      case 't2i_adapter':
        return data as T2IAdapterConfig;
      case 'ip_adapter':
        return data as IPAdapterConfig;
      case 'controlnet':
        return data as ControlNetConfig;
      case 'vae':
        return data as VAEConfig;
      default:
        return data as DiffusersModelConfig;
    }
  }, [data]);

  const {
    register,
    handleSubmit,
    control,
    formState: { errors },
    reset,
  } = useForm<AnyModelConfig>({
    defaultValues: {
      ...modelData,
    },
    mode: 'onChange',
  });

  const handleClickCancel = useCallback(() => {
    dispatch(setSelectedModelMode('view'));
  }, [dispatch]);

  if (isLoading) {
    return <Text>Loading</Text>;
  }

  if (!modelData) {
    return <Text>Something went wrong</Text>;
  }
  return (
    <Flex flexDir="column" h="full">
      <Flex w="full" justifyContent="space-between" gap={4} alignItems="center">
        <Input
          {...register('name', {
            validate: (value) => value.trim().length > 3 || 'Must be at least 3 characters',
          })}
          size="lg"
        />
        <Flex gap={2}>
          <Button size="sm" onClick={handleClickCancel}>
            Cancel
          </Button>
          <Button size="sm" colorScheme="invokeYellow">
            Save
          </Button>
        </Flex>
      </Flex>

      <Flex flexDir="column" gap={3} mt="4">
        <Flex>
          <FormControl flexDir="column" alignItems="flex-start" gap={1}>
            <FormLabel>Description</FormLabel>
            <Textarea fontSize="md" resize="none" {...register('description')} />
          </FormControl>
        </Flex>
        <Heading as="h3" fontSize="md" mt="4">
          Model Settings
        </Heading>
        <Flex gap={4}>
          <FormControl flexDir="column" alignItems="flex-start" gap={1}>
            <FormLabel>Base Model</FormLabel>
            <BaseModelSelect<AnyModelConfig> control={control} name="base" />
          </FormControl>
          <FormControl flexDir="column" alignItems="flex-start" gap={1}>
            <FormLabel>Model Type</FormLabel>
            <ModelTypeSelect<AnyModelConfig> control={control} name="type" />
          </FormControl>
        </Flex>
        <Flex gap={4}>
          <FormControl flexDir="column" alignItems="flex-start" gap={1}>
            <FormLabel>Format</FormLabel>
            <ModelFormatSelect<AnyModelConfig> control={control} name="format" />
          </FormControl>
          <FormControl flexDir="column" alignItems="flex-start" gap={1}>
            <FormLabel>Path</FormLabel>
            <Input
              {...register('path', {
                validate: (value) => value.trim().length > 0 || 'Must provide a path',
              })}
            />
          </FormControl>
        </Flex>
        {modelData.type === 'main' && (
          <>
            <Flex gap={4}>
              {modelData.format === 'diffusers' && (
                <FormControl flexDir="column" alignItems="flex-start" gap={1}>
                  <FormLabel>Repo Variant</FormLabel>
                  <RepoVariantSelect<AnyModelConfig> control={control} name="repo_variant" />
                </FormControl>
              )}
              {modelData.format === 'checkpoint' && (
                <FormControl flexDir="column" alignItems="flex-start" gap={1}>
                  <FormLabel>Config Path</FormLabel>
                  <Input {...register('config')} />
                </FormControl>
              )}

              <FormControl flexDir="column" alignItems="flex-start" gap={1}>
                <FormLabel>Variant</FormLabel>
                <ModelVariantSelect<AnyModelConfig> control={control} name="variant" />
              </FormControl>
            </Flex>
            <Flex gap={4}>
              <FormControl flexDir="column" alignItems="flex-start" gap={1}>
                <FormLabel>Prediction Type</FormLabel>
                <PredictionTypeSelect<AnyModelConfig> control={control} name="prediction_type" />
              </FormControl>
              <FormControl flexDir="column" alignItems="flex-start" gap={1}>
                <FormLabel>Upcast Attention</FormLabel>
                <BooleanSelect<AnyModelConfig> control={control} name="upcast_attention" />
              </FormControl>
            </Flex>
            <Flex gap={4}>
              <FormControl flexDir="column" alignItems="flex-start" gap={1}>
                <FormLabel>ZTSNR Training</FormLabel>
                <BooleanSelect<AnyModelConfig> control={control} name="ztsnr_training" />
              </FormControl>
              <FormControl flexDir="column" alignItems="flex-start" gap={1}>
                <FormLabel>VAE Path</FormLabel>
                <Input {...register('vae')} />
              </FormControl>
            </Flex>
          </>
        )}
        {modelData.type === 'ip_adapter' && (
          <Flex gap={4}>
            <FormControl flexDir="column" alignItems="flex-start" gap={1}>
              <FormLabel>Image Encoder Model ID</FormLabel>
              <Input {...register('image_encoder_model_id')} />
            </FormControl>
          </Flex>
        )}
      </Flex>
    </Flex>
  );
};
