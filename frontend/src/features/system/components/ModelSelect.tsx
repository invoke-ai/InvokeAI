import { Flex } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { requestModelChange } from 'app/socketio/actions';
import { useAppDispatch, useAppSelector } from 'app/store';
import IAISelect from 'common/components/IAISelect';
import _ from 'lodash';
import { ChangeEvent } from 'react';
import { systemSelector } from '../store/systemSelectors';

const selector = createSelector(
  [systemSelector],
  (system) => {
    const { isProcessing, model_list } = system;
    const models = _.map(model_list, (model, key) => key);
    const activeModel = _.reduce(
      model_list,
      (acc, model, key) => {
        if (model.status === 'active') {
          acc = key;
        }

        return acc;
      },
      ''
    );

    return { models, activeModel, isProcessing };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: _.isEqual,
    },
  }
);

const ModelSelect = () => {
  const dispatch = useAppDispatch();
  const { models, activeModel, isProcessing } = useAppSelector(selector);
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
        isDisabled={isProcessing}
        value={activeModel}
        validValues={models}
        onChange={handleChangeModel}
      />
    </Flex>
  );
};

export default ModelSelect;
