import { createSelector } from '@reduxjs/toolkit';
import { memo, useCallback } from 'react';
import { isEqual } from 'lodash-es';
import { useTranslation } from 'react-i18next';

import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  selectModelsAll,
  selectModelsById,
  selectModelsIds,
} from '../store/modelSlice';
import { RootState } from 'app/store/store';
import { modelSelected } from 'features/parameters/store/generationSlice';
import { generationSelector } from 'features/parameters/store/generationSelectors';
import IAICustomSelect, {
  ItemTooltips,
} from 'common/components/IAICustomSelect';

const selector = createSelector(
  [(state: RootState) => state, generationSelector],
  (state, generation) => {
    const selectedModel = selectModelsById(state, generation.model);
    const allModelNames = selectModelsIds(state).map((id) => String(id));
    const allModelTooltips = selectModelsAll(state).reduce(
      (allModelTooltips, model) => {
        allModelTooltips[model.name] = model.description ?? '';
        return allModelTooltips;
      },
      {} as ItemTooltips
    );
    return {
      allModelNames,
      allModelTooltips,
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
  const { allModelNames, allModelTooltips, selectedModel } =
    useAppSelector(selector);
  const handleChangeModel = useCallback(
    (v: string | null | undefined) => {
      if (!v) {
        return;
      }
      dispatch(modelSelected(v));
    },
    [dispatch]
  );

  return (
    <IAICustomSelect
      label={t('modelManager.model')}
      tooltip={selectedModel?.description}
      items={allModelNames}
      itemTooltips={allModelTooltips}
      selectedItem={selectedModel?.name ?? ''}
      setSelectedItem={handleChangeModel}
      withCheckIcon={true}
      tooltipProps={{ placement: 'top', hasArrow: true }}
    />
  );
};

export default memo(ModelSelect);
