import { Button, Flex, FormControl, FormLabel, Heading, Switch } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { useEncoderModelSettings } from 'features/modelManagerV2/hooks/useEncoderModelSettings';
import { selectSelectedModelKey } from 'features/modelManagerV2/store/modelManagerV2Slice';
import type { FormField } from 'features/modelManagerV2/subpanels/ModelPanel/MainModelDefaultSettings/MainModelDefaultSettings';
import { toast } from 'features/toast/toast';
import type { ChangeEvent } from 'react';
import { memo, useCallback, useMemo } from 'react';
import type { Control, SubmitHandler } from 'react-hook-form';
import { useController, useForm } from 'react-hook-form';
import { useTranslation } from 'react-i18next';
import { PiCheckBold } from 'react-icons/pi';
import { useUpdateModelMutation } from 'services/api/endpoints/models';
import type {
  CLIPEmbedModelConfig,
  CLIPVisionModelConfig,
  LlavaOnevisionModelConfig,
  Qwen3EncoderModelConfig,
  SigLIPModelConfig,
  T5EncoderModelConfig,
} from 'services/api/types';

export type EncoderModelSettingsFormData = {
  cpuOnly: FormField<boolean>;
};

type EncoderModelConfig =
  | CLIPEmbedModelConfig
  | T5EncoderModelConfig
  | Qwen3EncoderModelConfig
  | CLIPVisionModelConfig
  | SigLIPModelConfig
  | LlavaOnevisionModelConfig;

type Props = {
  modelConfig: EncoderModelConfig;
};

const DefaultCpuOnly = memo((props: { name: 'cpuOnly'; control: Control<EncoderModelSettingsFormData> }) => {
  const { field } = useController(props);
  const { t } = useTranslation();

  const onChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      const updatedValue = {
        ...(field.value as FormField<boolean>),
        value: e.target.checked,
        isEnabled: e.target.checked,
      };
      field.onChange(updatedValue);
    },
    [field]
  );

  const value = useMemo(() => {
    return (field.value as FormField<boolean>).value;
  }, [field.value]);

  return (
    <FormControl>
      <InformationalPopover feature="cpuOnly">
        <FormLabel>{t('modelManager.runOnCpu')}</FormLabel>
      </InformationalPopover>
      <Switch isChecked={value} onChange={onChange} />
    </FormControl>
  );
});

DefaultCpuOnly.displayName = 'DefaultCpuOnly';

export const EncoderModelSettings = memo(({ modelConfig }: Props) => {
  const selectedModelKey = useAppSelector(selectSelectedModelKey);
  const { t } = useTranslation();

  const settingsDefaults = useEncoderModelSettings(modelConfig);
  const [updateModel, { isLoading: isLoadingUpdateModel }] = useUpdateModelMutation();

  const { handleSubmit, control, formState, reset } = useForm<EncoderModelSettingsFormData>({
    defaultValues: settingsDefaults,
  });

  const onSubmit = useCallback<SubmitHandler<EncoderModelSettingsFormData>>(
    (data) => {
      if (!selectedModelKey) {
        return;
      }

      const body = {
        cpu_only: data.cpuOnly.isEnabled ? data.cpuOnly.value : null,
      };

      updateModel({
        key: selectedModelKey,
        body,
      })
        .unwrap()
        .then((_) => {
          toast({
            id: 'ENCODER_SETTINGS_SAVED',
            title: t('modelManager.settingsSaved'),
            status: 'success',
          });
          reset(data);
        })
        .catch((error) => {
          if (error) {
            toast({
              id: 'ENCODER_SETTINGS_SAVE_FAILED',
              title: `${error.data.detail} `,
              status: 'error',
            });
          }
        });
    },
    [selectedModelKey, reset, updateModel, t]
  );

  return (
    <>
      <Flex gap="4" justifyContent="space-between" w="full" pb={4}>
        <Heading fontSize="md">{t('modelManager.settings')}</Heading>
        <Button
          size="sm"
          leftIcon={<PiCheckBold />}
          colorScheme="invokeYellow"
          isDisabled={!formState.isDirty}
          onClick={handleSubmit(onSubmit)}
          isLoading={isLoadingUpdateModel}
        >
          {t('common.save')}
        </Button>
      </Flex>

      <DefaultCpuOnly control={control} name="cpuOnly" />
    </>
  );
});

EncoderModelSettings.displayName = 'EncoderModelSettings';
