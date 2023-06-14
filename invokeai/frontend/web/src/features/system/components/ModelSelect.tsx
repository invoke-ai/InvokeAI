import { createSelector } from '@reduxjs/toolkit';
import { isEqual } from 'lodash-es';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIMantineSelect, {
  IAISelectDataType,
} from 'common/components/IAIMantineSelect';
import { generationSelector } from 'features/parameters/store/generationSelectors';
import { modelSelected } from 'features/parameters/store/generationSlice';
import { selectModelsAll, selectModelsById } from '../store/modelSlice';

const selector = createSelector(
  [(state: RootState) => state, generationSelector],
  (state, generation) => {
    const selectedModel = selectModelsById(state, generation.model);

    const modelData = selectModelsAll(state)
      .map<IAISelectDataType>((m) => ({
        value: m.name,
        label: m.name,
      }))
      .sort((a, b) => a.label.localeCompare(b.label));
    return {
      selectedModel,
      modelData,
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
  const { selectedModel, modelData } = useAppSelector(selector);
  const handleChangeModel = useCallback(
    (v: string | null) => {
      if (!v) {
        return;
      }
      dispatch(modelSelected(v));
    },
    [dispatch]
  );

  return (
    <IAIMantineSelect
      tooltip={selectedModel?.description}
      label={t('modelManager.model')}
      value={selectedModel?.name ?? ''}
      placeholder="Pick one"
      data={modelData}
      onChange={handleChangeModel}
    />
  );
};

export default memo(ModelSelect);
