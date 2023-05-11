import { useTranslation } from 'react-i18next';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { ChangeEvent, memo, useCallback } from 'react';
import { createSelector } from '@reduxjs/toolkit';
import { generationSelector } from 'features/parameters/store/generationSelectors';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAISwitch from 'common/components/IAISwitch';
import { setSeamlessYAxis } from 'features/parameters/store/generationSlice';

const selector = createSelector(
  generationSelector,
  (generation) => {
    const { seamlessYAxis } = generation;

    return { seamlessYAxis };
  },
  defaultSelectorOptions
);

const ParamSeamlessYAxis = () => {
  const { t } = useTranslation();
  const { seamlessYAxis } = useAppSelector(selector);

  const dispatch = useAppDispatch();

  const handleChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      dispatch(setSeamlessYAxis(e.target.checked));
    },
    [dispatch]
  );

  return (
    <IAISwitch
      label={t('parameters.seamlessYAxis')}
      aria-label={t('parameters.seamlessYAxis')}
      isChecked={seamlessYAxis}
      onChange={handleChange}
    />
  );
};

export default memo(ParamSeamlessYAxis);
