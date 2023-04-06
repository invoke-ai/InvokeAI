import { Flex } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { ChangeEvent } from 'react';
import { isEqual, map } from 'lodash';
import { useTranslation } from 'react-i18next';

import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import IAISelect from 'common/components/IAISelect';
import { modelSelector } from '../store/modelSelectors';
import { setCurrentModel } from '../store/modelSlice';

const selector = createSelector(
  [modelSelector],
  (model) => {
    const { modelList, currentModel } = model;
    const models = map(modelList, (model, key) => key);
    return { models, currentModel, modelList };
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
  const { models, currentModel, modelList } = useAppSelector(selector);
  const handleChangeModel = (e: ChangeEvent<HTMLSelectElement>) => {
    dispatch(setCurrentModel(e.target.value));
  };
  const currentModelDescription =
    currentModel && modelList[currentModel].description;

  return (
    <Flex
      style={{
        paddingInlineStart: 1.5,
      }}
    >
      <IAISelect
        style={{ fontSize: 'sm' }}
        aria-label={t('accessibility.modelSelect')}
        tooltip={currentModelDescription}
        value={currentModel}
        validValues={models}
        onChange={handleChangeModel}
      />
    </Flex>
  );
};

export default ModelSelect;
