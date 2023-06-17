import { createSelector } from '@reduxjs/toolkit';
import { isEqual } from 'lodash-es';
import { memo, useCallback, useEffect } from 'react';
import { useTranslation } from 'react-i18next';

import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIMantineSelect, {
  IAISelectDataType,
} from 'common/components/IAIMantineSelect';
import { generationSelector } from 'features/parameters/store/generationSelectors';
import {
  modelSelected,
  setCurrentModelType,
} from 'features/parameters/store/generationSlice';

import {
  selectAllSD1Models,
  selectByIdSD1Models,
} from '../store/models/sd1ModelSlice';
import {
  selectAllSD2Models,
  selectByIdSD2Models,
} from '../store/models/sd2ModelSlice';

export const modelSelector = createSelector(
  [(state: RootState) => state, generationSelector],
  (state, generation) => {
    let selectedModel = selectByIdSD1Models(state, generation.model);
    if (selectedModel === undefined)
      selectedModel = selectByIdSD2Models(state, generation.model);

    const sd1ModelData = selectAllSD1Models(state)
      .map<IAISelectDataType>((m) => ({
        value: m.name,
        label: m.name,
        group: '1.x Models',
      }))
      .sort((a, b) => a.label.localeCompare(b.label));

    const sd2ModelData = selectAllSD2Models(state)
      .map<IAISelectDataType>((m) => ({
        value: m.name,
        label: m.name,
        group: '2.x Models',
      }))
      .sort((a, b) => a.label.localeCompare(b.label));

    return {
      selectedModel,
      sd1ModelData,
      sd2ModelData,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: isEqual,
    },
  }
);

export type ModelLoaderTypes = 'sd1_model_loader' | 'sd2_model_loader';

const MODEL_LOADER_MAP = {
  'sd-1': 'sd1_model_loader',
  'sd-2': 'sd2_model_loader',
};

const ModelSelect = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const { selectedModel, sd1ModelData, sd2ModelData } =
    useAppSelector(modelSelector);

  useEffect(() => {
    if (selectedModel)
      dispatch(
        setCurrentModelType(
          MODEL_LOADER_MAP[selectedModel?.base_model] as ModelLoaderTypes
        )
      );
  }, [dispatch, selectedModel]);

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
      data={sd1ModelData.concat(sd2ModelData)}
      onChange={handleChangeModel}
    />
  );
};

export default memo(ModelSelect);
