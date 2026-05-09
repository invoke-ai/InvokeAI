import { CompositeNumberInput, CompositeSlider, Flex, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { SettingToggle } from 'features/modelManagerV2/subpanels/ModelPanel/SettingToggle';
import { CONSTRAINTS } from 'features/parameters/components/Dimensions/DimensionsWidth';
import { memo, useCallback, useMemo } from 'react';
import type { UseControllerProps } from 'react-hook-form';
import { useController } from 'react-hook-form';
import { useTranslation } from 'react-i18next';

import type { MainModelDefaultSettingsFormData } from './MainModelDefaultSettings';

type DefaultWidthType = MainModelDefaultSettingsFormData['width'];

type Props = {
  control: UseControllerProps<MainModelDefaultSettingsFormData>['control'];
  optimalDimension: number;
};

export const DefaultWidth = memo(({ control, optimalDimension }: Props) => {
  const { field } = useController({ control, name: 'width' });
  const { t } = useTranslation();
  const marks = useMemo(() => [CONSTRAINTS.sliderMin, optimalDimension, CONSTRAINTS.sliderMax], [optimalDimension]);

  const onChange = useCallback(
    (v: number) => {
      const updatedValue = {
        ...(field.value as DefaultWidthType),
        value: v,
      };
      field.onChange(updatedValue);
    },
    [field]
  );

  const value = useMemo(() => {
    return (field.value as DefaultWidthType).value;
  }, [field.value]);

  const isDisabled = useMemo(() => {
    return !(field.value as DefaultWidthType).isEnabled;
  }, [field.value]);

  return (
    <FormControl flexDir="column" gap={2} alignItems="flex-start">
      <Flex justifyContent="space-between" w="full">
        <InformationalPopover feature="paramWidth">
          <FormLabel>{t('parameters.width')}</FormLabel>
        </InformationalPopover>
        <SettingToggle control={control} name="width" />
      </Flex>

      <Flex w="full" gap={4}>
        <CompositeSlider
          value={value}
          min={CONSTRAINTS.sliderMin}
          max={CONSTRAINTS.sliderMax}
          step={CONSTRAINTS.coarseStep}
          fineStep={CONSTRAINTS.fineStep}
          onChange={onChange}
          marks={marks}
          isDisabled={isDisabled}
        />
        <CompositeNumberInput
          value={value}
          min={CONSTRAINTS.numberInputMin}
          max={CONSTRAINTS.numberInputMax}
          step={CONSTRAINTS.coarseStep}
          fineStep={CONSTRAINTS.fineStep}
          onChange={onChange}
          isDisabled={isDisabled}
        />
      </Flex>
    </FormControl>
  );
});

DefaultWidth.displayName = 'DefaultWidth';
