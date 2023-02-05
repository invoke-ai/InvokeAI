import { createSelector } from '@reduxjs/toolkit';
import _ from 'lodash';
import React from 'react';
import type { RootState } from 'app/store';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import IAINumberInput from 'common/components/IAINumberInput';
import {
  OptionsState,
  setIterations,
} from 'features/options/store/optionsSlice';
import { useTranslation } from 'react-i18next';

const mainIterationsSelector = createSelector(
  [(state: RootState) => state.options],
  (options: OptionsState) => {
    const { iterations } = options;

    return {
      iterations,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: _.isEqual,
    },
  }
);

export default function MainIterations() {
  const dispatch = useAppDispatch();
  const { iterations } = useAppSelector(mainIterationsSelector);
  const { t } = useTranslation();

  const handleChangeIterations = (v: number) => dispatch(setIterations(v));

  return (
    <IAINumberInput
      label={t('options:images')}
      step={1}
      min={1}
      max={9999}
      onChange={handleChangeIterations}
      value={iterations}
      width="auto"
      labelFontSize={0.5}
      styleClass="main-option-block"
      textAlign="center"
    />
  );
}
