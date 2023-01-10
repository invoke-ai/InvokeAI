import { useState } from 'react';
import {
  Container,
  Flex,
  Modal,
  ModalCloseButton,
  ModalContent,
  ModalHeader,
  ModalOverlay,
  SimpleGrid,
  Text,
  useDisclosure,
} from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { requestModelChange } from 'app/socketio/actions';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import _ from 'lodash';
import { systemSelector } from '../store/systemSelectors';
import { useTranslation } from 'react-i18next';
import ModelGridItem from './ModelGridItem';
import IAIInput from 'common/components/IAIInput';
import type { ChangeEvent } from 'react';

const modelGridSelector = createSelector(
  [systemSelector],
  (system) => {
    const models = _.map(system.model_list, (model, key) => {
      return { name: key, ...model };
    });
    const activeModel = models.find(m => m.status === 'active');
    const { isProcessing } = system;

    return { models, activeModel, isProcessing };
  }
);

const ModelGrid = ({value, models, onChange}) => {
  const [searchText, setSearchText] = useState<string>('');

  const { t } = useTranslation();

  const handleSearchFilter = _.debounce((e: ChangeEvent<HTMLInputElement>) => {
    setSearchText(e.target.value.toLowerCase());
  }, 400);

  return (
    <>
      <Container padding="1.5rem">
        <IAIInput
          onChange={handleSearchFilter}
          label={t('modelmanager:search')}
        />
      </Container>
      <SimpleGrid 
        columns={3} spacing={4}
        overflowY='scroll'
        padding='1.5rem'
        maxH='70vh'
      >
        {models.filter(model => {
          if (searchText) 
            return model.name.toLowerCase().startsWith(searchText);
          return true;
        })
        .map((model, i) => 
          <ModelGridItem
            key={i}
            model={model}
            isSelected={value.name === model.name}
            onSelect={onChange}
          />
        )}
      </SimpleGrid>
    </>
  )
}


const ModelSelect = () => {
  const dispatch = useAppDispatch();
  const { models, activeModel, isProcessing } =
    useAppSelector(modelGridSelector);
  const handleChangeModel = (modelName: string) => {
    dispatch(requestModelChange(modelName));
    onModelGridModalClose()
  };

  const {
    isOpen: isModelGridModalOpen,
    onOpen: onModelGridModalOpen,
    onClose: onModelGridModalClose,
  } = useDisclosure();

  const { t } = useTranslation();

  return (
    <Flex
      style={{
        paddingLeft: '0.3rem',
      }}
    >
      <Text
        fontSize='sm'
        onClick={onModelGridModalOpen}
        cursor={isProcessing ? 'not-allowed' : 'pointer'}
        color={isProcessing ? 'var(--subtext-color-bright)' : 'var(--text-color)'}
      >
        <b>{activeModel && activeModel.name}</b>
      </Text>
      <Modal
        isOpen={!isProcessing && isModelGridModalOpen}
        onClose={onModelGridModalClose}
        size="6xl"
      >
        <ModalOverlay />
        <ModalContent className=" modal">
          <ModalCloseButton className="modal-close-btn" />
          <ModalHeader>{t('selectModel')}</ModalHeader>
            <ModelGrid
              value={activeModel}
              models={models}
              onChange={handleChangeModel}
            />
        </ModalContent>
      </Modal>
    </Flex>
  );
};

export default ModelSelect;
