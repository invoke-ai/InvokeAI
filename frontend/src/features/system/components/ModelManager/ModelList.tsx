import { useState } from 'react';
import { Flex, Text } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import IAIInput from 'common/components/IAIInput';

import AddModel from './AddModel';
import ModelListItem from './ModelListItem';

import { useAppSelector } from 'app/storeHooks';
import { useTranslation } from 'react-i18next';

import _ from 'lodash';

import type { ChangeEvent, ReactNode } from 'react';
import type { RootState } from 'app/store';
import type { SystemState } from 'features/system/store/systemSlice';

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

  const [searchText, setSearchText] = useState<string>('');

  const { t } = useTranslation();

  const handleSearchFilter = _.debounce((e: ChangeEvent<HTMLInputElement>) => {
    setSearchText(e.target.value);
  }, 400);

  const renderModelListItems = () => {
    const modelListItemsToRender: ReactNode[] = [];
    const filteredModelListItemsToRender: ReactNode[] = [];

    models.forEach((model, i) => {
      if (model.name.startsWith(searchText)) {
        filteredModelListItemsToRender.push(
          <ModelListItem
            key={i}
            name={model.name}
            status={model.status}
            description={model.description}
          />
        );
      }
      modelListItemsToRender.push(
        <ModelListItem
          key={i}
          name={model.name}
          status={model.status}
          description={model.description}
        />
      );
    });

    return searchText !== ''
      ? filteredModelListItemsToRender
      : modelListItemsToRender;
  };

  return (
    <Flex flexDirection={'column'} rowGap="2rem" width="50%" minWidth="50%">
      <Flex justifyContent={'space-between'}>
        <Text fontSize={'1.4rem'} fontWeight="bold">
          {t('modelmanager:availableModels')}
        </Text>
        <AddModel />
      </Flex>

      <IAIInput
        onChange={handleSearchFilter}
        label={t('modelmanager:search')}
      />

      <Flex
        flexDirection={'column'}
        gap={1}
        maxHeight={window.innerHeight - 360}
        overflow={'scroll'}
        paddingRight="1rem"
      >
        {renderModelListItems()}
      </Flex>
    </Flex>
  );
};

export default ModelList;
