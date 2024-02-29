import { Button, Flex, Heading } from '@invoke-ai/ui-library';
import { SubmitHandler, useForm } from 'react-hook-form';
import { SettingToggle } from './DefaultSettings/SettingToggle';
import { DefaultCfgScale } from './DefaultSettings/DefaultCfgScale';
import { DefaultSteps } from './DefaultSettings/DefaultSteps';
import { useAppSelector } from '../../../../app/store/storeHooks';
import { DefaultScheduler } from './DefaultSettings/DefaultScheduler';
import { selectGenerationSlice } from '../../../parameters/store/generationSlice';
import { createMemoizedSelector } from '../../../../app/store/createMemoizedSelector';
import { selectConfigSlice } from '../../../system/store/configSlice';
import { DefaultVaePrecision } from './DefaultSettings/DefaultVaePrecision';
import { DefaultCfgRescaleMultiplier } from './DefaultSettings/DefaultCfgRescaleMultiplier';
import { DefaultVae } from './DefaultSettings/DefaultVae';
import { t } from 'i18next';
import { IoPencil } from 'react-icons/io5';
import { each } from 'lodash-es';

export interface FormField<T> {
  value: T;
  isEnabled: boolean;
}

export type DefaultSettingsFormData = {
  vae: FormField<string | null>;
  vaePrecision: FormField<string>;
  scheduler: FormField<string>;
  steps: FormField<number>;
  cfgScale: FormField<number>;
  cfgRescaleMultiplier: FormField<number>;
};

const initialStatesSelector = createMemoizedSelector(selectConfigSlice, (config) => {
  const { steps, guidance, scheduler, cfgRescaleMultiplier, vaePrecision } = config.sd;

  return {
    initialSteps: steps.initial,
    initialCfg: guidance.initial,
    initialScheduler: scheduler,
    initialCfgRescaleMultiplier: cfgRescaleMultiplier.initial,
    initialVaePrecision: vaePrecision,
  };
});

export const DefaultSettings = () => {
  const { initialSteps, initialCfg, initialScheduler, initialCfgRescaleMultiplier, initialVaePrecision } =
    useAppSelector(initialStatesSelector);

  const { handleSubmit, control, formState } = useForm<DefaultSettingsFormData>({
    defaultValues: {
      vae: { isEnabled: false, value: null },
      vaePrecision: { isEnabled: false, value: initialVaePrecision },
      scheduler: { isEnabled: false, value: initialScheduler },
      steps: { isEnabled: false, value: initialSteps },
      cfgScale: { isEnabled: false, value: initialCfg },
      cfgRescaleMultiplier: { isEnabled: false, value: initialCfgRescaleMultiplier },
    },
  });

  const onSubmit: SubmitHandler<DefaultSettingsFormData> = (data) => {
    const body: { [key: string]: string | number | null } = {};
    each(data, (value, key) => {
      if (value.isEnabled) {
        body[key] = value.value;
      }
    });
    console.log(body);
  };

  return (
    <>
      <Flex gap="2" justifyContent="space-between" w="full" mb={5}>
        <Heading fontSize="md">Default Settings</Heading>
        <Button
          size="sm"
          leftIcon={<IoPencil />}
          colorScheme="invokeYellow"
          isDisabled={!formState.isDirty}
          onClick={handleSubmit(onSubmit)}
          type="submit"
        >
          {t('common.save')}
        </Button>
      </Flex>

      <Flex flexDir="column" gap={8}>
        <Flex gap={8}>
          <Flex gap={4} w="full">
            <SettingToggle control={control} name="vae" />
            <DefaultVae control={control} name="vae" />
          </Flex>
          <Flex gap={4} w="full">
            <SettingToggle control={control} name="vaePrecision" />
            <DefaultVaePrecision control={control} name="vaePrecision" />
          </Flex>
        </Flex>
        <Flex gap={8}>
          <Flex gap={4} w="full">
            <SettingToggle control={control} name="scheduler" />
            <DefaultScheduler control={control} name="scheduler" />
          </Flex>
          <Flex gap={4} w="full">
            <SettingToggle control={control} name="steps" />
            <DefaultSteps control={control} name="steps" />
          </Flex>
        </Flex>
        <Flex gap={8}>
          <Flex gap={4} w="full">
            <SettingToggle control={control} name="cfgScale" />
            <DefaultCfgScale control={control} name="cfgScale" />
          </Flex>
          <Flex gap={4} w="full">
            <SettingToggle control={control} name="cfgRescaleMultiplier" />
            <DefaultCfgRescaleMultiplier control={control} name="cfgRescaleMultiplier" />
          </Flex>
        </Flex>
      </Flex>
    </>
  );
};
