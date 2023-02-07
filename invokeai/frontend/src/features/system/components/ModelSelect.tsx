import { Flex } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { requestModelChange } from 'app/socketio/actions';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import IAISelect from 'common/components/IAISelect';
import { isEqual, map } from 'lodash';

import { ChangeEvent } from 'react';
import { activeModelSelector, systemSelector } from '../store/systemSelectors';

const selector = createSelector(
  [systemSelector],
  (system) => {
    const { isProcessing, model_list } = system;
    const models = map(model_list, (model, key) => key);
    return { models, isProcessing };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: isEqual,
    },
  }
);

const ModelSelect = () => {
  const dispatch = useAppDispatch();
  const { models, isProcessing } = useAppSelector(selector);
  const activeModel = useAppSelector(activeModelSelector);
  const handleChangeModel = (e: ChangeEvent<HTMLSelectElement>) => {
    dispatch(requestModelChange(e.target.value));
  };

  return (
    <Flex
      style={{
        paddingLeft: '0.3rem',
      }}
    >
      <IAISelect
        style={{ fontSize: '0.8rem' }}
        tooltip={activeModel.description}
        isDisabled={isProcessing}
        value={activeModel.name}
        validValues={models}
        onChange={handleChangeModel}
      />
    </Flex>
  );
};

export default ModelSelect;
