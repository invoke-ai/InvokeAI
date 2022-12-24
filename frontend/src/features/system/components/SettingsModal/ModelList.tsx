import {
  Button,
  Tooltip,
  Spacer,
  Accordion,
  AccordionItem,
  AccordionButton,
  AccordionPanel,
  AccordionIcon,
  Text,
} from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import _ from 'lodash';
import { ModelStatus } from 'app/invokeai';
import { requestModelChange } from 'app/socketio/actions';
import { RootState } from 'app/store';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import { SystemState } from 'features/system/store/systemSlice';
import { useTranslation } from 'react-i18next';

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
  const { t } = useTranslation();
  return (
    <div className="model-list-item">
      <Tooltip label={description} hasArrow placement="bottom">
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
          isDisabled={status === 'active' || isProcessing || !isConnected}
        >
          {t('common:load')}
        </Button>
      </div>
    </div>
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
  const { t } = useTranslation();

  return (
    <Accordion
      allowToggle
      className="model-list-accordion"
      variant={'unstyled'}
    >
      <AccordionItem>
        <AccordionButton>
          <div className="model-list-button">
            <Text
              fontSize="sm"
              fontWeight="bold"
              color="var(--text-color-secondary)"
            >
              {t('settings:models')}
            </Text>
            <AccordionIcon />
          </div>
        </AccordionButton>

        <AccordionPanel>
          <div className="model-list-list">
            {models.map((model, i) => (
              <ModelListItem
                key={i}
                name={model.name}
                status={model.status}
                description={model.description}
              />
            ))}
          </div>
        </AccordionPanel>
      </AccordionItem>
    </Accordion>
  );
};

export default ModelList;
