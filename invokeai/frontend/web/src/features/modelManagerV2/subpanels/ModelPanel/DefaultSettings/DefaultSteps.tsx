import { CompositeNumberInput, CompositeSlider, Flex,FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { useCallback,useMemo  } from 'react';
import type {UseControllerProps } from 'react-hook-form';
import { useController } from 'react-hook-form';
import { useTranslation } from 'react-i18next';

import type { DefaultSettingsFormData } from './DefaultSettingsForm';

type DefaultSteps = DefaultSettingsFormData['steps'];

export function DefaultSteps(props: UseControllerProps<DefaultSettingsFormData>) {
  const { field } = useController(props);

  const sliderMin = useAppSelector((s) => s.config.sd.steps.sliderMin);
  const sliderMax = useAppSelector((s) => s.config.sd.steps.sliderMax);
  const numberInputMin = useAppSelector((s) => s.config.sd.steps.numberInputMin);
  const numberInputMax = useAppSelector((s) => s.config.sd.steps.numberInputMax);
  const coarseStep = useAppSelector((s) => s.config.sd.steps.coarseStep);
  const fineStep = useAppSelector((s) => s.config.sd.steps.fineStep);
  const { t } = useTranslation();
  const marks = useMemo(() => [sliderMin, Math.floor(sliderMax / 2), sliderMax], [sliderMax, sliderMin]);

  const onChange = useCallback(
    (v: number) => {
      const updatedValue = {
        ...(field.value as DefaultSteps),
        value: v,
      };
      field.onChange(updatedValue);
    },
    [field]
  );

  const value = useMemo(() => {
    return (field.value as DefaultSteps).value;
  }, [field.value]);

  const isDisabled = useMemo(() => {
    return !(field.value as DefaultSteps).isEnabled;
  }, [field.value]);

  return (
    <FormControl flexDir="column" gap={1} alignItems="flex-start">
      <InformationalPopover feature="paramSteps">
        <FormLabel>{t('parameters.steps')}</FormLabel>
      </InformationalPopover>
      <Flex w="full" gap={1}>
        <CompositeSlider
          value={value}
          min={sliderMin}
          max={sliderMax}
          step={coarseStep}
          fineStep={fineStep}
          onChange={onChange}
          marks={marks}
          isDisabled={isDisabled}
        />
        <CompositeNumberInput
          value={value}
          min={numberInputMin}
          max={numberInputMax}
          step={coarseStep}
          fineStep={fineStep}
          onChange={onChange}
          isDisabled={isDisabled}
        />
      </Flex>
    </FormControl>
  );
}
