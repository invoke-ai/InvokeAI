import { CompositeNumberInput, CompositeSlider, Flex, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { SettingToggle } from 'features/modelManagerV2/subpanels/ModelPanel/SettingToggle';
import { selectHeightConfig } from 'features/system/store/configSlice';
import { memo, useCallback, useMemo } from 'react';
import type { UseControllerProps } from 'react-hook-form';
import { useController } from 'react-hook-form';
import { useTranslation } from 'react-i18next';

import type { MainModelDefaultSettingsFormData } from './MainModelDefaultSettings';

type DefaultHeightType = MainModelDefaultSettingsFormData['height'];

type Props = {
  control: UseControllerProps<MainModelDefaultSettingsFormData>['control'];
  optimalDimension: number;
};

export const DefaultHeight = memo(({ control, optimalDimension }: Props) => {
  const { field } = useController({ control, name: 'height' });
  const config = useAppSelector(selectHeightConfig);
  const { t } = useTranslation();
  const marks = useMemo(
    () => [config.sliderMin, optimalDimension, config.sliderMax],
    [config.sliderMin, optimalDimension, config.sliderMax]
  );

  const onChange = useCallback(
    (v: number) => {
      const updatedValue = {
        ...(field.value as DefaultHeightType),
        value: v,
      };
      field.onChange(updatedValue);
    },
    [field]
  );

  const value = useMemo(() => {
    return field.value.value;
  }, [field.value]);

  const isDisabled = useMemo(() => {
    return !field.value.isEnabled;
  }, [field.value]);

  return (
    <FormControl flexDir="column" gap={2} alignItems="flex-start">
      <Flex justifyContent="space-between" w="full">
        <InformationalPopover feature="paramHeight">
          <FormLabel>{t('parameters.height')}</FormLabel>
        </InformationalPopover>
        <SettingToggle control={control} name="height" />
      </Flex>

      <Flex w="full" gap={4}>
        <CompositeSlider
          value={value}
          min={config.sliderMin}
          max={config.sliderMax}
          step={config.coarseStep}
          fineStep={config.fineStep}
          onChange={onChange}
          marks={marks}
          isDisabled={isDisabled}
        />
        <CompositeNumberInput
          value={value}
          min={config.numberInputMin}
          max={config.numberInputMax}
          step={config.coarseStep}
          fineStep={config.fineStep}
          onChange={onChange}
          isDisabled={isDisabled}
        />
      </Flex>
    </FormControl>
  );
});

DefaultHeight.displayName = 'DefaultHeight';
