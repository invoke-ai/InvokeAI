import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvSlider } from 'common/components/InvSlider/InvSlider';
import { maxPromptsChanged } from 'features/dynamicPrompts/store/dynamicPromptsSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const ParamDynamicPromptsMaxPrompts = () => {
  const maxPrompts = useAppSelector((s) => s.dynamicPrompts.maxPrompts);
  const min = useAppSelector((s) => s.config.sd.dynamicPrompts.maxPrompts.min);
  const sliderMax = useAppSelector(
    (s) => s.config.sd.dynamicPrompts.maxPrompts.sliderMax
  );
  const inputMax = useAppSelector(
    (s) => s.config.sd.dynamicPrompts.maxPrompts.inputMax
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
        min={min}
        max={sliderMax}
        value={maxPrompts}
        defaultValue={initial}
        onChange={handleChange}
        marks
        withNumberInput
        numberInputMax={inputMax}
      />
    </InvControl>
  );
};

export default memo(ParamDynamicPromptsMaxPrompts);
