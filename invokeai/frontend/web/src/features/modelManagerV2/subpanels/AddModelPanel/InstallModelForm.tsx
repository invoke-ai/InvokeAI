import { Button, Checkbox, Flex, FormControl, FormHelperText, FormLabel, Input } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useInstallModel } from 'features/modelManagerV2/hooks/useInstallModel';
import {
  selectShouldInstallInPlace,
  shouldInstallInPlaceChanged,
} from 'features/modelManagerV2/store/modelManagerV2Slice';
import { t } from 'i18next';
import type { ChangeEvent } from 'react';
import { memo, useCallback, useState } from 'react';
import type { SubmitHandler } from 'react-hook-form';
import { useForm } from 'react-hook-form';

import { ManualModelConfigPanel } from './ManualModelConfigPanel';

type SimpleImportModelConfig = {
  location: string;
};

export const InstallModelForm = memo(() => {
  const inplace = useAppSelector(selectShouldInstallInPlace);
  const dispatch = useAppDispatch();
  const [installModel, { isLoading }] = useInstallModel();
  const [isManualConfig, setIsManualConfig] = useState(false);
  const [manualConfig, setManualConfig] = useState({});

  const { register, handleSubmit, formState, reset } = useForm<SimpleImportModelConfig>({
    defaultValues: {
      location: '',
    },
    mode: 'onChange',
  });

  const resetForm = useCallback(() => reset(undefined, { keepValues: true }), [reset]);

  const onSubmit = useCallback<SubmitHandler<SimpleImportModelConfig>>(
    (values) => {
      if (!values?.location) {
        return;
      }

      installModel({
        source: values.location,
        inplace: inplace,
        config: isManualConfig ? manualConfig : {},
        onSuccess: resetForm,
        onError: resetForm,
      });
    },
    [installModel, resetForm, inplace, isManualConfig, manualConfig]
  );

  const onChangeInplace = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      dispatch(shouldInstallInPlaceChanged(e.target.checked));
    },
    [dispatch]
  );

  const onChangeManualConfig = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    setIsManualConfig(e.target.checked);
    if (!e.target.checked) {
      setManualConfig({});
    }
  }, []);

  return (
    <form onSubmit={handleSubmit(onSubmit)}>
      <Flex flexDir="column" gap={4}>
        <FormControl orientation="vertical">
          <FormLabel>{t('modelManager.urlOrLocalPath')}</FormLabel>
          <Flex alignItems="center" gap={3} w="full">
            <Input placeholder={t('modelManager.simpleModelPlaceholder')} {...register('location')} />
            <Button
              onClick={handleSubmit(onSubmit)}
              isDisabled={!formState.dirtyFields.location}
              isLoading={isLoading}
              size="sm"
            >
              {t('modelManager.install')}
            </Button>
          </Flex>
          <FormHelperText>{t('modelManager.urlOrLocalPathHelper')}</FormHelperText>
        </FormControl>

        <FormControl>
          <Flex flexDir="column" gap={2}>
            <Flex gap={4}>
              <Checkbox isChecked={inplace} onChange={onChangeInplace} />
              <FormLabel>
                {t('modelManager.inplaceInstall')} ({t('modelManager.localOnly')})
              </FormLabel>
            </Flex>
            <FormHelperText>{t('modelManager.inplaceInstallDesc')}</FormHelperText>
          </Flex>
        </FormControl>

        <FormControl>
          <Flex flexDir="column" gap={2}>
            <Flex gap={4}>
              <Checkbox isChecked={isManualConfig} onChange={onChangeManualConfig} />
              <FormLabel>{t('modelManager.manualConfiguration')}</FormLabel>
            </Flex>
            <FormHelperText>{t('modelManager.manualConfigurationDesc')}</FormHelperText>
          </Flex>
        </FormControl>

        {isManualConfig && <ManualModelConfigPanel config={manualConfig} onChange={setManualConfig} />}
      </Flex>
    </form>
  );
});

InstallModelForm.displayName = 'InstallModelForm';
