import { createSelector } from '@reduxjs/toolkit';
import type { RootState } from 'app/store';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import IAINumberInput from 'common/components/IAINumberInput';
import {
  GenerationState,
  setIterations,
} from 'features/parameters/store/generationSlice';
import { isEqual } from 'lodash';

import { useTranslation } from 'react-i18next';

const mainIterationsSelector = createSelector(
  [(state: RootState) => state.generation],
  (parameters: GenerationState) => {
    const { iterations } = parameters;

    return {
      iterations,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: isEqual,
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
      label={t('parameters:images')}
      step={1}
      min={1}
      max={9999}
      onChange={handleChangeIterations}
      value={iterations}
      width="auto"
      labelFontSize={0.5}
      styleClass="main-settings-block"
      textAlign="center"
    />
  );
}
