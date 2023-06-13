import { createSelector } from '@reduxjs/toolkit';
import { ChangeEvent, memo, useCallback } from 'react';
import { isEqual } from 'lodash-es';
import { useTranslation } from 'react-i18next';

import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { selectModelsAll, selectModelsById } from '../store/modelSlice';
import { RootState } from 'app/store/store';
import { modelSelected } from 'features/parameters/store/generationSlice';
import { generationSelector } from 'features/parameters/store/generationSelectors';
import IAICustomSelect, {
  IAICustomSelectOption,
} from 'common/components/IAICustomSelect';
import IAISelect from 'common/components/IAISelect';

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
    <IAISelect
      label={t('modelManager.model')}
      tooltip={selectedModel?.description}
      validValues={modelData}
      value={selectedModel?.name ?? ''}
      onChange={handleChangeModel}
      tooltipProps={{ placement: 'top', hasArrow: true }}
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
