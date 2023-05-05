import { createSelector } from '@reduxjs/toolkit';
import { ChangeEvent, memo } from 'react';
import { isEqual } from 'lodash-es';
import { useTranslation } from 'react-i18next';

import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAISelect from 'common/components/IAISelect';
import {
  // modelSelected,
  selectedModelSelector,
  selectModelsById,
  selectModelsIds,
} from '../store/modelSlice';
import { RootState } from 'app/store/store';
import generationSlice, {
  modelSelected,
} from 'features/parameters/store/generationSlice';
import { generationSelector } from 'features/parameters/store/generationSelectors';

const selector = createSelector(
  [(state: RootState) => state, generationSelector],
  (state, generation) => {
    // const selectedModel = selectedModelSelector(state);
    const selectedModel = selectModelsById(state, generation.model);
    const allModelNames = selectModelsIds(state);
    return {
      allModelNames,
      selectedModel,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: isEqual,
    },
  }
);

const ModelSelect = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const { allModelNames, selectedModel } = useAppSelector(selector);
  const handleChangeModel = (e: ChangeEvent<HTMLSelectElement>) => {
    dispatch(modelSelected(e.target.value));
  };

  return (
    <IAISelect
      label={t('modelManager.model')}
      style={{ fontSize: 'sm' }}
      aria-label={t('accessibility.modelSelect')}
      tooltip={selectedModel?.description || ''}
      value={selectedModel?.name || undefined}
      validValues={allModelNames}
      onChange={handleChangeModel}
    />
  );
};

export default memo(ModelSelect);
