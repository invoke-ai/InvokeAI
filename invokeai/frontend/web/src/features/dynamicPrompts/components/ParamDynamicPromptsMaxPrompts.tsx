import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvSlider } from 'common/components/InvSlider/InvSlider';
import { maxPromptsChanged } from 'features/dynamicPrompts/store/dynamicPromptsSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const ParamDynamicPromptsMaxPrompts = () => {
  const maxPrompts = useAppSelector((s) => s.dynamicPrompts.maxPrompts);
  const sliderMin = useAppSelector(
    (s) => s.config.sd.dynamicPrompts.maxPrompts.sliderMin
  );
  const sliderMax = useAppSelector(
    (s) => s.config.sd.dynamicPrompts.maxPrompts.sliderMax
  );
  const numberInputMin = useAppSelector(
    (s) => s.config.sd.dynamicPrompts.maxPrompts.numberInputMin
  );
  const numberInputMax = useAppSelector(
    (s) => s.config.sd.dynamicPrompts.maxPrompts.numberInputMax
  );
  const initial = useAppSelector(
    (s) => s.config.sd.dynamicPrompts.maxPrompts.initial
  );
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
    <InvControl
      label={t('dynamicPrompts.maxPrompts')}
      isDisabled={isDisabled}
      feature="dynamicPromptsMaxPrompts"
      renderInfoPopoverInPortal={false}
    >
      <InvSlider
        min={sliderMin}
        max={sliderMax}
        value={maxPrompts}
        defaultValue={initial}
        onChange={handleChange}
        marks
        withNumberInput
        numberInputMin={numberInputMin}
        numberInputMax={numberInputMax}
      />
    </InvControl>
  );
};

export default memo(ParamDynamicPromptsMaxPrompts);
