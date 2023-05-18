import { useTranslation } from 'react-i18next';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { ChangeEvent, memo, useCallback } from 'react';
import { createSelector } from '@reduxjs/toolkit';
import { generationSelector } from 'features/parameters/store/generationSelectors';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAISwitch from 'common/components/IAISwitch';
import { setSeamlessXAxis } from 'features/parameters/store/generationSlice';

const selector = createSelector(
  generationSelector,
  (generation) => {
    const { seamlessXAxis } = generation;

    return { seamlessXAxis };
  },
  defaultSelectorOptions
);

const ParamSeamlessXAxis = () => {
  const { t } = useTranslation();
  const { seamlessXAxis } = useAppSelector(selector);

  const dispatch = useAppDispatch();

  const handleChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      dispatch(setSeamlessXAxis(e.target.checked));
    },
    [dispatch]
  );

  return (
    <IAISwitch
      label={t('parameters.seamlessXAxis')}
      aria-label={t('parameters.seamlessXAxis')}
      isChecked={seamlessXAxis}
      onChange={handleChange}
    />
  );
};

export default memo(ParamSeamlessXAxis);
