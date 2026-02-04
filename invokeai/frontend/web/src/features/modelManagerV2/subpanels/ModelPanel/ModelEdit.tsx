import {
  Button,
  Checkbox,
  Flex,
  FormControl,
  FormErrorMessage,
  FormLabel,
  Heading,
  Input,
  SimpleGrid,
  Text,
  Textarea,
} from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { setSelectedModelMode } from 'features/modelManagerV2/store/modelManagerV2Slice';
import { ModelHeader } from 'features/modelManagerV2/subpanels/ModelPanel/ModelHeader';
import { toast } from 'features/toast/toast';
import { memo, useCallback, useMemo } from 'react';
import { type SubmitHandler, useForm, useWatch } from 'react-hook-form';
import { useTranslation } from 'react-i18next';
import { PiCheckBold, PiXBold } from 'react-icons/pi';
import { type UpdateModelArg, useUpdateModelMutation } from 'services/api/endpoints/models';
import {
  type AnyModelConfig,
  type ExternalModelCapabilities,
  isExternalApiModelConfig,
  type UpdateModelBody,
} from 'services/api/types';

import BaseModelSelect from './Fields/BaseModelSelect';
import ModelFormatSelect from './Fields/ModelFormatSelect';
import ModelTypeSelect from './Fields/ModelTypeSelect';
import ModelVariantSelect from './Fields/ModelVariantSelect';
import PredictionTypeSelect from './Fields/PredictionTypeSelect';
import { ModelFooter } from './ModelFooter';

type Props = {
  modelConfig: AnyModelConfig;
};

type ModelEditFormValues = UpdateModelBody;

const stringFieldOptions = {
  validate: (value?: string | null) => (value && value.trim().length > 3) || 'Must be at least 3 characters',
};

export const ModelEdit = memo(({ modelConfig }: Props) => {
  const { t } = useTranslation();
  const [updateModel, { isLoading: isSubmitting }] = useUpdateModelMutation();
  const dispatch = useAppDispatch();
  const isExternal = useMemo(() => isExternalApiModelConfig(modelConfig), [modelConfig]);

  const form = useForm<ModelEditFormValues>({
    defaultValues: modelConfig,
    mode: 'onChange',
  });

  const externalModes = useWatch({
    control: form.control,
    name: 'capabilities.modes',
  }) as ExternalModelCapabilities['modes'] | undefined;

  const modeSet = useMemo(() => new Set(externalModes ?? []), [externalModes]);

  const toggleMode = useCallback(
    (mode: ExternalModelCapabilities['modes'][number]) => {
      const nextModes = modeSet.has(mode)
        ? externalModes?.filter((value) => value !== mode)
        : [...(externalModes ?? []), mode];
      form.setValue('capabilities.modes', nextModes ?? [], { shouldDirty: true, shouldValidate: true });
    },
    [externalModes, form, modeSet]
  );

  const handleToggleTxt2Img = useCallback(() => toggleMode('txt2img'), [toggleMode]);
  const handleToggleImg2Img = useCallback(() => toggleMode('img2img'), [toggleMode]);
  const handleToggleInpaint = useCallback(() => toggleMode('inpaint'), [toggleMode]);

  const parseOptionalNumber = useCallback((value: string | null | undefined) => {
    if (value === null || value === undefined || value === '') {
      return null;
    }
    if (typeof value !== 'string') {
      return Number.isNaN(Number(value)) ? null : Number(value);
    }
    if (value.trim() === '') {
      return null;
    }
    const parsed = Number(value);
    return Number.isNaN(parsed) ? null : parsed;
  }, []);

  const onSubmit = useCallback<SubmitHandler<ModelEditFormValues>>(
    (values) => {
      const responseBody: UpdateModelArg = {
        key: modelConfig.key,
        body: values,
      };
      updateModel(responseBody)
        .unwrap()
        .then((payload) => {
          form.reset(payload, { keepDefaultValues: true });
          dispatch(setSelectedModelMode('view'));
          toast({
            id: 'MODEL_UPDATED',
            title: t('modelManager.modelUpdated'),
            status: 'success',
          });
        })
        .catch((_) => {
          form.reset();
          toast({
            id: 'MODEL_UPDATE_FAILED',
            title: t('modelManager.modelUpdateFailed'),
            status: 'error',
          });
        });
    },
    [dispatch, modelConfig.key, form, t, updateModel]
  );

  const handleClickCancel = useCallback(() => {
    dispatch(setSelectedModelMode('view'));
  }, [dispatch]);

  return (
    <Flex flexDir="column" gap={4}>
      <ModelHeader modelConfig={modelConfig}>
        <Button flexShrink={0} size="sm" onClick={handleClickCancel} leftIcon={<PiXBold />}>
          {t('common.cancel')}
        </Button>
        <Button
          flexShrink={0}
          size="sm"
          colorScheme="invokeYellow"
          leftIcon={<PiCheckBold />}
          onClick={form.handleSubmit(onSubmit)}
          isLoading={isSubmitting}
          isDisabled={Boolean(Object.keys(form.formState.errors).length)}
        >
          {t('common.save')}
        </Button>
      </ModelHeader>
      <Flex flexDir="column" h="full">
        <form>
          <Flex w="full" justifyContent="space-between" gap={4} alignItems="center">
            <FormControl
              flexDir="column"
              alignItems="flex-start"
              gap={1}
              isInvalid={Boolean(form.formState.errors.name)}
            >
              <FormLabel>{t('modelManager.modelName')}</FormLabel>
              <Input {...form.register('name', stringFieldOptions)} size="md" />

              {form.formState.errors.name?.message && (
                <FormErrorMessage>{form.formState.errors.name?.message}</FormErrorMessage>
              )}
            </FormControl>
          </Flex>

          <Flex flexDir="column" gap={3} mt="4">
            <Flex gap="4" alignItems="center">
              <FormControl flexDir="column" alignItems="flex-start" gap={1}>
                <FormLabel>{t('modelManager.description')}</FormLabel>
                <Textarea {...form.register('description')} minH={32} />
              </FormControl>
            </Flex>
            <Heading as="h3" fontSize="md" mt="4">
              {t('modelManager.modelSettings')}
            </Heading>
            <Text variant="subtext" color="warning.300">
              {t('modelManager.modelSettingsWarning')}
            </Text>
            <SimpleGrid columns={2} gap={4}>
              <FormControl flexDir="column" alignItems="flex-start" gap={1}>
                <FormLabel>{t('modelManager.modelType')}</FormLabel>
                <ModelTypeSelect control={form.control} />
              </FormControl>
              <FormControl flexDir="column" alignItems="flex-start" gap={1}>
                <FormLabel>{t('modelManager.modelFormat')}</FormLabel>
                <ModelFormatSelect control={form.control} />
              </FormControl>
              <FormControl flexDir="column" alignItems="flex-start" gap={1}>
                <FormLabel>{t('modelManager.baseModel')}</FormLabel>
                <BaseModelSelect control={form.control} />
              </FormControl>
              <FormControl flexDir="column" alignItems="flex-start" gap={1}>
                <FormLabel>{t('modelManager.variant')}</FormLabel>
                <ModelVariantSelect control={form.control} />
              </FormControl>
              <FormControl flexDir="column" alignItems="flex-start" gap={1}>
                <FormLabel>{t('modelManager.pathToConfig')}</FormLabel>
                <Input {...form.register('config_path')} />
              </FormControl>
              <FormControl flexDir="column" alignItems="flex-start" gap={1}>
                <FormLabel>{t('modelManager.predictionType')}</FormLabel>
                <PredictionTypeSelect control={form.control} />
              </FormControl>
              <FormControl flexDir="column" alignItems="flex-start" gap={1}>
                <FormLabel>{t('modelManager.upcastAttention')}</FormLabel>
                <Checkbox {...form.register('upcast_attention')} />
              </FormControl>
            </SimpleGrid>
            {isExternal && (
              <>
                <Heading as="h3" fontSize="md" mt="4">
                  {t('modelManager.externalProvider')}
                </Heading>
                <SimpleGrid columns={2} gap={4}>
                  <FormControl flexDir="column" alignItems="flex-start" gap={1}>
                    <FormLabel>{t('modelManager.providerId')}</FormLabel>
                    <Input {...form.register('provider_id', stringFieldOptions)} size="md" />
                  </FormControl>
                  <FormControl flexDir="column" alignItems="flex-start" gap={1}>
                    <FormLabel>{t('modelManager.providerModelId')}</FormLabel>
                    <Input {...form.register('provider_model_id', stringFieldOptions)} size="md" />
                  </FormControl>
                </SimpleGrid>
                <Heading as="h3" fontSize="md" mt="4">
                  {t('modelManager.externalCapabilities')}
                </Heading>
                <SimpleGrid columns={2} gap={4}>
                  <FormControl flexDir="column" alignItems="flex-start" gap={1}>
                    <FormLabel>{t('modelManager.supportedModes')}</FormLabel>
                    <Flex gap={3} wrap="wrap">
                      <Checkbox isChecked={modeSet.has('txt2img')} onChange={handleToggleTxt2Img}>
                        txt2img
                      </Checkbox>
                      <Checkbox isChecked={modeSet.has('img2img')} onChange={handleToggleImg2Img}>
                        img2img
                      </Checkbox>
                      <Checkbox isChecked={modeSet.has('inpaint')} onChange={handleToggleInpaint}>
                        inpaint
                      </Checkbox>
                    </Flex>
                  </FormControl>
                  <FormControl flexDir="column" alignItems="flex-start" gap={1}>
                    <FormLabel>{t('modelManager.supportsNegativePrompt')}</FormLabel>
                    <Checkbox {...form.register('capabilities.supports_negative_prompt')} />
                  </FormControl>
                  <FormControl flexDir="column" alignItems="flex-start" gap={1}>
                    <FormLabel>{t('modelManager.supportsReferenceImages')}</FormLabel>
                    <Checkbox {...form.register('capabilities.supports_reference_images')} />
                  </FormControl>
                  <FormControl flexDir="column" alignItems="flex-start" gap={1}>
                    <FormLabel>{t('modelManager.supportsSeed')}</FormLabel>
                    <Checkbox {...form.register('capabilities.supports_seed')} />
                  </FormControl>
                  <FormControl flexDir="column" alignItems="flex-start" gap={1}>
                    <FormLabel>{t('modelManager.supportsGuidance')}</FormLabel>
                    <Checkbox {...form.register('capabilities.supports_guidance')} />
                  </FormControl>
                  <FormControl flexDir="column" alignItems="flex-start" gap={1}>
                    <FormLabel>{t('modelManager.maxImagesPerRequest')}</FormLabel>
                    <Input
                      type="number"
                      {...form.register('capabilities.max_images_per_request', {
                        setValueAs: parseOptionalNumber,
                      })}
                    />
                  </FormControl>
                  <FormControl flexDir="column" alignItems="flex-start" gap={1}>
                    <FormLabel>{t('modelManager.maxReferenceImages')}</FormLabel>
                    <Input
                      type="number"
                      {...form.register('capabilities.max_reference_images', {
                        setValueAs: parseOptionalNumber,
                      })}
                    />
                  </FormControl>
                  <FormControl flexDir="column" alignItems="flex-start" gap={1}>
                    <FormLabel>{t('modelManager.maxImageWidth')}</FormLabel>
                    <Input
                      type="number"
                      {...form.register('capabilities.max_image_size.width', {
                        setValueAs: parseOptionalNumber,
                      })}
                    />
                  </FormControl>
                  <FormControl flexDir="column" alignItems="flex-start" gap={1}>
                    <FormLabel>{t('modelManager.maxImageHeight')}</FormLabel>
                    <Input
                      type="number"
                      {...form.register('capabilities.max_image_size.height', {
                        setValueAs: parseOptionalNumber,
                      })}
                    />
                  </FormControl>
                </SimpleGrid>
                <Heading as="h3" fontSize="md" mt="4">
                  {t('modelManager.externalDefaults')}
                </Heading>
                <SimpleGrid columns={2} gap={4}>
                  <FormControl flexDir="column" alignItems="flex-start" gap={1}>
                    <FormLabel>{t('modelManager.width')}</FormLabel>
                    <Input
                      type="number"
                      {...form.register('default_settings.width', {
                        setValueAs: parseOptionalNumber,
                      })}
                    />
                  </FormControl>
                  <FormControl flexDir="column" alignItems="flex-start" gap={1}>
                    <FormLabel>{t('modelManager.height')}</FormLabel>
                    <Input
                      type="number"
                      {...form.register('default_settings.height', {
                        setValueAs: parseOptionalNumber,
                      })}
                    />
                  </FormControl>
                  <FormControl flexDir="column" alignItems="flex-start" gap={1}>
                    <FormLabel>{t('parameters.steps')}</FormLabel>
                    <Input
                      type="number"
                      {...form.register('default_settings.steps', {
                        setValueAs: parseOptionalNumber,
                      })}
                    />
                  </FormControl>
                  <FormControl flexDir="column" alignItems="flex-start" gap={1}>
                    <FormLabel>{t('parameters.guidance')}</FormLabel>
                    <Input
                      type="number"
                      {...form.register('default_settings.guidance', {
                        setValueAs: parseOptionalNumber,
                      })}
                    />
                  </FormControl>
                  <FormControl flexDir="column" alignItems="flex-start" gap={1}>
                    <FormLabel>{t('modelManager.numImages')}</FormLabel>
                    <Input
                      type="number"
                      {...form.register('default_settings.num_images', {
                        setValueAs: parseOptionalNumber,
                      })}
                    />
                  </FormControl>
                </SimpleGrid>
              </>
            )}
          </Flex>
        </form>
      </Flex>
      <ModelFooter modelConfig={modelConfig} isEditing={true} />
    </Flex>
  );
});

ModelEdit.displayName = 'ModelEdit';
