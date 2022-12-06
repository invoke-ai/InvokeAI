import {
  Button,
  Tooltip,
  Spacer,
  IconButton,
  Flex,
  Text,
} from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import _ from 'lodash';
import { ModelStatus } from 'app/invokeai';
import { deleteModel, requestModelChange } from 'app/socketio/actions';
import { RootState, useAppDispatch, useAppSelector } from 'app/store';
import { SystemState } from 'features/system/store/systemSlice';
import AddModel from './AddModel';
import { DeleteIcon } from '@chakra-ui/icons';
import IAIAlertDialog from 'common/components/IAIAlertDialog';

type ModelListItemProps = {
  name: string;
  status: ModelStatus;
  description: string;
};

const ModelListItem = (props: ModelListItemProps) => {
  const { isProcessing, isConnected } = useAppSelector(
    (state: RootState) => state.system
  );

  const dispatch = useAppDispatch();
  const { name, status, description } = props;

  const handleChangeModel = () => {
    dispatch(requestModelChange(name));
  };

  const handleModelDelete = () => {
    dispatch(deleteModel(name));
  };

  const statusTextColor = () => {
    switch (status) {
      case 'active':
        return 'var(--status-good-color)';
      case 'cached':
        return 'var(--status-working-color)';
      case 'not loaded':
        return 'var(--text-color-secondary)';
    }
  };

  return (
    <Flex alignItems={'center'}>
      <Tooltip label={description} hasArrow placement="bottom">
        <Text>{name}</Text>
      </Tooltip>
      <Spacer />

      <Flex gap={4}>
        <Text color={statusTextColor()}>{status}</Text>

        <Button
          size={'sm'}
          onClick={handleChangeModel}
          isDisabled={status === 'active' || isProcessing || !isConnected}
          className="modal-close-btn"
        >
          Load
        </Button>
        <IAIAlertDialog
          title={'Delete Model?'}
          acceptCallback={handleModelDelete}
          acceptButtonText={'Delete'}
          triggerComponent={
            <IconButton
              icon={<DeleteIcon />}
              size={'sm'}
              aria-label="Delete Config"
              isDisabled={status === 'active' || isProcessing || !isConnected}
              className=" modal-close-btn"
            />
          }
        >
          <Flex rowGap={'1rem'} flexDirection="column">
            <p style={{ fontWeight: 'bold' }}>
              Are you sure you want to delete this model entry from InvokeAI?
            </p>
            <p style={{ color: 'var(--text-color-secondary' }}>
              This will <strong>not</strong> delete the model checkpoint file
              from your disk. You can readd them if you wish to.
            </p>
          </Flex>
        </IAIAlertDialog>
      </Flex>
    </Flex>
  );
};

const modelListSelector = createSelector(
  (state: RootState) => state.system,
  (system: SystemState) => {
    const models = _.map(system.model_list, (model, key) => {
      return { name: key, ...model };
    });

    const activeModel = models.find((model) => model.status === 'active');

    return {
      models,
      activeModel: activeModel,
    };
  }
);

const ModelList = () => {
  const { models } = useAppSelector(modelListSelector);

  return (
    <Flex flexDirection={'column'} rowGap="1rem">
      <Flex justifyContent={'space-between'}>
        <Text fontSize={'1.4rem'} fontWeight="bold">
          Available Models
        </Text>
        <AddModel />
      </Flex>

      <Flex flexDirection={'column'} rowGap={2}>
        {models.map((model, i) => (
          <ModelListItem
            key={i}
            name={model.name}
            status={model.status}
            description={model.description}
          />
        ))}
      </Flex>
    </Flex>
  );
};

export default ModelList;
