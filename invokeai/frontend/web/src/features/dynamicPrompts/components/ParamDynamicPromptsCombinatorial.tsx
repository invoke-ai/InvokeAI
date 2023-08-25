import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAISwitch from 'common/components/IAISwitch';
import { memo, useCallback } from 'react';
import { combinatorialToggled } from '../store/dynamicPromptsSlice';

const selector = createSelector(
  stateSelector,
  (state) => {
    const { combinatorial, isEnabled } = state.dynamicPrompts;

    return { combinatorial, isDisabled: !isEnabled };
  },
  defaultSelectorOptions
);

const ParamDynamicPromptsCombinatorial = () => {
  const { combinatorial, isDisabled } = useAppSelector(selector);
  const dispatch = useAppDispatch();

  const handleChange = useCallback(() => {
    dispatch(combinatorialToggled());
  }, [dispatch]);

  return (
    <IAISwitch
      isDisabled={isDisabled}
      label="Combinatorial Generation"
      isChecked={combinatorial}
      onChange={handleChange}
    />
  );
};

export default memo(ParamDynamicPromptsCombinatorial);
