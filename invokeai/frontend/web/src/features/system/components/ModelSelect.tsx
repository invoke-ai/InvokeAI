import { Flex } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { ChangeEvent } from 'react';
import { isEqual } from 'lodash';
import { useTranslation } from 'react-i18next';

import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import IAISelect from 'common/components/IAISelect';
import {
  modelSelected,
  selectedModelSelector,
  selectModelsIds,
} from '../store/modelSlice';
import { RootState } from 'app/store';

const selector = createSelector(
  [(state: RootState) => state],
  (state) => {
    const selectedModel = selectedModelSelector(state);
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
    <Flex
      style={{
        paddingInlineStart: 1.5,
      }}
    >
      <IAISelect
        style={{ fontSize: 'sm' }}
        aria-label={t('accessibility.modelSelect')}
        tooltip={selectedModel?.description || ''}
        value={selectedModel?.name || undefined}
        validValues={allModelNames}
        onChange={handleChangeModel}
      />
    </Flex>
  );
};

export default ModelSelect;
