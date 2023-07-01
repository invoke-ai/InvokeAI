import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { combinatorialToggled } from '../store/slice';
import { createSelector } from '@reduxjs/toolkit';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { useCallback } from 'react';
import { stateSelector } from 'app/store/store';
import IAISwitch from 'common/components/IAISwitch';

const selector = createSelector(
  stateSelector,
  (state) => {
    const { combinatorial } = state.dynamicPrompts;

    return { combinatorial };
  },
  defaultSelectorOptions
);

const ParamDynamicPromptsCombinatorial = () => {
  const { combinatorial } = useAppSelector(selector);
  const dispatch = useAppDispatch();

  const handleChange = useCallback(() => {
    dispatch(combinatorialToggled());
  }, [dispatch]);

  return (
    <IAISwitch
      label="Combinatorial Generation"
      isChecked={combinatorial}
      onChange={handleChange}
    />
  );
};

export default ParamDynamicPromptsCombinatorial;
