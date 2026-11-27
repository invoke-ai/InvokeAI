import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import {
  maxPromptsChanged,
  selectDynamicPromptsCombinatorial,
  selectDynamicPromptsMaxPrompts,
} from 'features/dynamicPrompts/store/dynamicPromptsSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const CONSTRAINTS = {
  initial: 100,
  sliderMin: 1,
  sliderMax: 1000,
  numberInputMin: 1,
  numberInputMax: 10000,
  fineStep: 1,
  coarseStep: 10,
};

const ParamDynamicPromptsMaxPrompts = () => {
  const maxPrompts = useAppSelector(selectDynamicPromptsMaxPrompts);
  const combinatorial = useAppSelector(selectDynamicPromptsCombinatorial);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const handleChange = useCallback(
    (v: number) => {
      dispatch(maxPromptsChanged(v));
    },
    [dispatch]
  );

  return (
    <FormControl isDisabled={!combinatorial}>
      <InformationalPopover feature="dynamicPromptsMaxPrompts" inPortal={false}>
        <FormLabel>{t('dynamicPrompts.maxPrompts')}</FormLabel>
      </InformationalPopover>
      <CompositeSlider
        min={CONSTRAINTS.sliderMin}
        max={CONSTRAINTS.sliderMax}
        value={maxPrompts}
        defaultValue={CONSTRAINTS.initial}
        onChange={handleChange}
        marks
      />
      <CompositeNumberInput
        min={CONSTRAINTS.numberInputMin}
        max={CONSTRAINTS.numberInputMax}
        value={maxPrompts}
        defaultValue={CONSTRAINTS.initial}
        onChange={handleChange}
      />
    </FormControl>
  );
};

export default memo(ParamDynamicPromptsMaxPrompts);
