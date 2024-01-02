import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvSlider } from 'common/components/InvSlider/InvSlider';
import { maxPromptsChanged } from 'features/dynamicPrompts/store/dynamicPromptsSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const selector = createMemoizedSelector(stateSelector, (state) => {
  const { maxPrompts, combinatorial } = state.dynamicPrompts;
  const { min, sliderMax, inputMax, initial } =
    state.config.sd.dynamicPrompts.maxPrompts;

  return {
    maxPrompts,
    min,
    sliderMax,
    inputMax,
    initial,
    isDisabled: !combinatorial,
  };
});

const ParamDynamicPromptsMaxPrompts = () => {
  const { maxPrompts, min, sliderMax, inputMax, initial, isDisabled } =
    useAppSelector(selector);
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
