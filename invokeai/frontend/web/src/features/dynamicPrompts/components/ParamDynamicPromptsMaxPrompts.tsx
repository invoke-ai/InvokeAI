import { CompositeNumberInput, CompositeSlider, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import {
  maxPromptsChanged,
  selectDynamicPromptsCombinatorial,
  selectDynamicPromptsMaxPrompts,
} from 'features/dynamicPrompts/store/dynamicPromptsSlice';
import { selectMaxPromptsConfig } from 'features/system/store/configSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const ParamDynamicPromptsMaxPrompts = () => {
  const maxPrompts = useAppSelector(selectDynamicPromptsMaxPrompts);
  const config = useAppSelector(selectMaxPromptsConfig);
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
