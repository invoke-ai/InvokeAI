import React, { useState, useTransition, useMemo } from 'react';
import { Box, Flex, Text } from '@chakra-ui/react';
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
    return models;
  },
  {
    memoizeOptions: {
      resultEqualityCheck: _.isEqual,
    },
  }
);

const ModelList = () => {
  const models = useAppSelector(modelListSelector);

  const [searchText, setSearchText] = useState<string>('');
  const [_, startTransition] = useTransition();

  const { t } = useTranslation();

  const handleSearchFilter = (e: ChangeEvent<HTMLInputElement>) => {
    startTransition(() => {
      setSearchText(e.target.value);
    });
  };

  const renderModelListItems = useMemo(() => {
    const ckptModelListItemsToRender: ReactNode[] = [];
    const diffusersModelListItemsToRender: ReactNode[] = [];
    const filteredModelListItemsToRender: ReactNode[] = [];

    models.forEach((model, i) => {
      if (model.name.toLowerCase().includes(searchText.toLowerCase())) {
        filteredModelListItemsToRender.push(
          <ModelListItem
            key={i}
            name={model.name}
            status={model.status}
            description={model.description}
          />
        );
      }
      if (model.format !== 'diffusers') {
        ckptModelListItemsToRender.push(
          <ModelListItem
            key={i}
            name={model.name}
            status={model.status}
            description={model.description}
          />
        );
      } else {
        diffusersModelListItemsToRender.push(
          <ModelListItem
            key={i}
            name={model.name}
            status={model.status}
            description={model.description}
          />
        );
      }
    });

    return searchText !== '' ? (
      filteredModelListItemsToRender
    ) : (
      <Flex flexDirection="column" rowGap="1.5rem">
        <Box>
          <Text
            fontWeight="bold"
            backgroundColor="var(--background-color)"
            padding="0.5rem 1rem"
            borderRadius="0.5rem"
            marginBottom="0.5rem"
            width="max-content"
          >
            Checkpoint Models
          </Text>
          {ckptModelListItemsToRender}
        </Box>
        <Box>
          <Text
            fontWeight="bold"
            backgroundColor="var(--background-color)"
            padding="0.5rem 1rem"
            borderRadius="0.5rem"
            marginBottom="0.5rem"
            width="max-content"
          >
            Diffusers Models
          </Text>
          {diffusersModelListItemsToRender}
        </Box>
      </Flex>
    );
  }, [models, searchText]);

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
        {renderModelListItems}
      </Flex>
    </Flex>
  );
};

export default ModelList;
