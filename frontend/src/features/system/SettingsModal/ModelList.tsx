import { Button, Tooltip, Spacer, Heading } from '@chakra-ui/react';
import _ from 'lodash';
import { Model, ModelStatus } from '../../../app/invokeai';
import { requestModelChange } from '../../../app/socketio/actions';
import { RootState, useAppDispatch, useAppSelector } from '../../../app/store';

type ModelListItemProps = {
  name: string;
  status: ModelStatus;
  description: string;
};

const ModelListItem = (props: ModelListItemProps) => {
  const dispatch = useAppDispatch();
  const { name, status, description } = props;
  const handleChangeModel = () => {
    dispatch(requestModelChange(name));
  };
  return (
    <div className="model-list-item">
      <Tooltip label={description} hasArrow placement="top">
        <div className="model-list-item-name">{name}</div>
      </Tooltip>
      <Spacer />
      <div className={`model-list-item-status ${status.split(' ').join('-')}`}>
        {status}
      </div>
      <div className="model-list-item-load-btn">
        <Button
          size={'sm'}
          onClick={handleChangeModel}
          isDisabled={status === 'active'}
        >
          Load
        </Button>
      </div>
    </div>
  );
};

const ModelList = () => {
  const { model_list } = useAppSelector((state: RootState) => state.system);

  return (
    <div className="model-list">
      <Heading size={'md'} className="model-list-header">
        Available Models
      </Heading>
      <div className="model-list-list">
        {_.map(model_list, (model: Model, key) => (
          <ModelListItem
            key={key}
            name={key}
            status={model.status}
            description={model.description}
          />
        ))}
      </div>
    </div>
  );
};

export default ModelList;
