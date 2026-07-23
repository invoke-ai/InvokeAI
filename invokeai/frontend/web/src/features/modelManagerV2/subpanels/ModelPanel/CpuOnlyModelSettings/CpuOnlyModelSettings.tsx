import { Button, Flex, FormControl, FormLabel, Heading, Switch } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import type { Feature } from 'common/components/InformationalPopover/constants';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import type { CpuOnlyModelSettingsFormData } from 'features/modelManagerV2/hooks/useCpuOnlyModelSettings';
import { useCpuOnlyModelSettings } from 'features/modelManagerV2/hooks/useCpuOnlyModelSettings';
import { selectSelectedModelKey } from 'features/modelManagerV2/store/modelManagerV2Slice';
import type { FormField } from 'features/modelManagerV2/subpanels/ModelPanel/MainModelDefaultSettings/MainModelDefaultSettings';
import { toast } from 'features/toast/toast';
import type { ChangeEvent } from 'react';
import { memo, useCallback, useEffect, useMemo } from 'react';
import type { Control, SubmitHandler } from 'react-hook-form';
import { useController, useForm } from 'react-hook-form';
import { useTranslation } from 'react-i18next';
import { PiCheckBold } from 'react-icons/pi';
import { useUpdateModelMutation } from 'services/api/endpoints/models';

const CpuOnlyToggle = memo(
  (props: { name: 'cpuOnly'; control: Control<CpuOnlyModelSettingsFormData>; feature: Feature; label: string }) => {
    const { field } = useController({ name: props.name, control: props.control });
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
        <InformationalPopover feature={props.feature}>
          <FormLabel>{t(props.label)}</FormLabel>
        </InformationalPopover>
        <Switch isChecked={value} onChange={onChange} />
      </FormControl>
    );
  }
);

CpuOnlyToggle.displayName = 'CpuOnlyToggle';

type Props = {
  modelConfig: { cpu_only?: boolean | null };
  /** Popover feature key describing the CPU-only behavior for this model type. */
  feature: Feature;
  /** i18n key for the toggle label. */
  label: string;
  /** Prefix for the save success/failure toast ids, e.g. `ENCODER_SETTINGS` or `VAE_SETTINGS`. */
  toastIdBase: string;
};

/**
 * Settings panel exposing the standalone `cpu_only` toggle. Shared by every model type that carries
 * a `cpu_only` config field (text encoders and VAEs) — the only per-type differences are the label,
 * popover copy, and toast ids, which are passed in as props.
 */
export const CpuOnlyModelSettings = memo(({ modelConfig, feature, label, toastIdBase }: Props) => {
  const selectedModelKey = useAppSelector(selectSelectedModelKey);
  const { t } = useTranslation();

  const settingsDefaults = useCpuOnlyModelSettings(modelConfig);
  const [updateModel, { isLoading: isLoadingUpdateModel }] = useUpdateModelMutation();

  const { handleSubmit, control, formState, reset } = useForm<CpuOnlyModelSettingsFormData>({
    defaultValues: settingsDefaults,
  });

  useEffect(() => {
    reset(settingsDefaults);
  }, [settingsDefaults, reset]);

  const onSubmit = useCallback<SubmitHandler<CpuOnlyModelSettingsFormData>>(
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
            id: `${toastIdBase}_SAVED`,
            title: t('modelManager.settingsSaved'),
            status: 'success',
          });
          reset(data);
        })
        .catch((error) => {
          if (error) {
            toast({
              id: `${toastIdBase}_SAVE_FAILED`,
              title: `${error.data.detail} `,
              status: 'error',
            });
          }
        });
    },
    [selectedModelKey, updateModel, toastIdBase, t, reset]
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

      <CpuOnlyToggle control={control} name="cpuOnly" feature={feature} label={label} />
    </>
  );
});

CpuOnlyModelSettings.displayName = 'CpuOnlyModelSettings';
