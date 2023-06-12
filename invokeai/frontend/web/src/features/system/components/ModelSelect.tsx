import { createSelector } from '@reduxjs/toolkit';
import { isEqual } from 'lodash-es';
import { ChangeEvent, memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import { Select } from '@mantine/core';
import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { IAICustomSelectOption } from 'common/components/IAICustomSelect';
import { generationSelector } from 'features/parameters/store/generationSelectors';
import { modelSelected } from 'features/parameters/store/generationSlice';
import { selectModelsAll, selectModelsById } from '../store/modelSlice';

const selector = createSelector(
  [(state: RootState) => state, generationSelector],
  (state, generation) => {
    const selectedModel = selectModelsById(state, generation.model);

    const modelData = selectModelsAll(state)
      .map((m) => ({
        value: m.name,
        key: m.name,
      }))
      .sort((a, b) => a.key.localeCompare(b.key));
    // const modelData = selectModelsAll(state)
    //   .map<IAICustomSelectOption>((m) => ({
    //     value: m.name,
    //     label: m.name,
    //     tooltip: m.description,
    //   }))
    //   .sort((a, b) => a.label.localeCompare(b.label));
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
    (e: ChangeEvent<HTMLSelectElement>) => {
      dispatch(modelSelected(e.target.value));
    },
    [dispatch]
  );
  // const handleChangeModel = useCallback(
  //   (v: string | null | undefined) => {
  //     if (!v) {
  //       return;
  //     }
  //     dispatch(modelSelected(v));
  //   },
  //   [dispatch]
  // );

  return (
    <Select
      label={t('modelManager.model')}
      value={selectedModel?.name ?? ''}
      placeholder="Pick one"
      data={modelData}
      searchable
      onChange={handleChangeModel}
    />
  );

  // return (
  //   <IAICustomSelect
  //     label={t('modelManager.model')}
  //     tooltip={selectedModel?.description}
  //     data={modelData}
  //     value={selectedModel?.name ?? ''}
  //     onChange={handleChangeModel}
  //     withCheckIcon={true}
  //     tooltipProps={{ placement: 'top', hasArrow: true }}
  //   />
  // );
};

export default memo(ModelSelect);
