import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import { maxPromptsChanged, selectDynamicPromptsSlice } from 'features/dynamicPrompts/store/dynamicPromptsSlice';
import { selectConfigSlice } from 'features/system/store/configSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const selectMaxPrompts = createSelector(selectDynamicPromptsSlice, (dynamicPrompts) => dynamicPrompts.maxPrompts);
const selectMaxPromptsConfig = createSelector(selectConfigSlice, (config) => config.sd.dynamicPrompts.maxPrompts);
const selectIsDisabled = createSelector(selectDynamicPromptsSlice, (dynamicPrompts) => !dynamicPrompts.combinatorial);

const ParamDynamicPromptsMaxPrompts = () => {
  const maxPrompts = useAppSelector(selectMaxPrompts);
  const config = useAppSelector(selectMaxPromptsConfig);
  const isDisabled = useAppSelector(selectIsDisabled);
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
        min={config.sliderMin}
        max={config.sliderMax}
        value={maxPrompts}
        defaultValue={config.initial}
        onChange={handleChange}
        marks
      />
      <CompositeNumberInput
        min={config.numberInputMin}
        max={config.numberInputMax}
        value={maxPrompts}
        defaultValue={config.initial}
        onChange={handleChange}
      />
    </FormControl>
  );
};

export default memo(ParamDynamicPromptsMaxPrompts);
