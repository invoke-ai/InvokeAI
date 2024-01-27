import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { maxPromptsChanged } from 'features/dynamicPrompts/store/dynamicPromptsSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const ParamDynamicPromptsMaxPrompts = () => {
  const maxPrompts = useAppSelector((s) => s.dynamicPrompts.maxPrompts);
  const sliderMin = useAppSelector((s) => s.config.sd.dynamicPrompts.maxPrompts.sliderMin);
  const sliderMax = useAppSelector((s) => s.config.sd.dynamicPrompts.maxPrompts.sliderMax);
  const numberInputMin = useAppSelector((s) => s.config.sd.dynamicPrompts.maxPrompts.numberInputMin);
  const numberInputMax = useAppSelector((s) => s.config.sd.dynamicPrompts.maxPrompts.numberInputMax);
  const initial = useAppSelector((s) => s.config.sd.dynamicPrompts.maxPrompts.initial);
  const isDisabled = useAppSelector((s) => !s.dynamicPrompts.combinatorial);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const handleChange = useCallback(
    (v: number) => {
      dispatch(maxPromptsChanged(v));
    },
    [dispatch]
  );

  return (
    <FormControl isDisabled={isDisabled}>
      <InformationalPopover feature="dynamicPromptsMaxPrompts" inPortal={false}>
        <FormLabel>{t('dynamicPrompts.maxPrompts')}</FormLabel>
      </InformationalPopover>
      <CompositeSlider
        min={sliderMin}
        max={sliderMax}
        value={maxPrompts}
        defaultValue={initial}
        onChange={handleChange}
        marks
      />
      <CompositeNumberInput
        min={numberInputMin}
        max={numberInputMax}
        value={maxPrompts}
        defaultValue={initial}
        onChange={handleChange}
      />
    </FormControl>
  );
};

export default memo(ParamDynamicPromptsMaxPrompts);
