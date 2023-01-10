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
import IAIButton from 'common/components/IAIButton';

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

function ModelFilterButton({
  label,
  isActive,
  onClick,
}: {
  label: string;
  isActive: boolean;
  onClick: () => void;
}) {
  return (
    <IAIButton
      onClick={onClick}
      isActive={isActive}
      _active={{
        backgroundColor: 'var(--accent-color)',
        _hover: { backgroundColor: 'var(--accent-color)' },
      }}
      size="sm"
    >
      {label}
    </IAIButton>
  );
}

const ModelList = () => {
  const models = useAppSelector(modelListSelector);

  const [searchText, setSearchText] = useState<string>('');
  const [isSelectedFilter, setIsSelectedFilter] = useState<
    'all' | 'ckpt' | 'diffusers'
  >('all');
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
    const localFilteredModelListItemsToRender: ReactNode[] = [];

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
        if (model.format === isSelectedFilter) {
          localFilteredModelListItemsToRender.push(
            <ModelListItem
              key={i}
              name={model.name}
              status={model.status}
              description={model.description}
            />
          );
        }
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
      isSelectedFilter === 'all' ? (
        <Box marginTop="1rem">{filteredModelListItemsToRender}</Box>
      ) : (
        <Box marginTop="1rem">{localFilteredModelListItemsToRender}</Box>
      )
    ) : (
      <Flex flexDirection="column" rowGap="1.5rem">
        {isSelectedFilter === 'all' && (
          <>
            <Box>
              <Text
                fontWeight="bold"
                backgroundColor="var(--background-color)"
                padding="0.5rem 1rem"
                borderRadius="0.5rem"
                margin="1rem 0"
                width="max-content"
                fontSize="14"
              >
                {t('modelmanager:checkpointModels')}
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
                fontSize="14"
              >
                {t('modelmanager:diffusersModels')}
              </Text>
              {diffusersModelListItemsToRender}
            </Box>
          </>
        )}

        {isSelectedFilter === 'ckpt' && (
          <Flex flexDirection="column" marginTop="1rem">
            {ckptModelListItemsToRender}
          </Flex>
        )}

        {isSelectedFilter === 'diffusers' && (
          <Flex flexDirection="column" marginTop="1rem">
            {diffusersModelListItemsToRender}
          </Flex>
        )}
      </Flex>
    );
  }, [models, searchText, t, isSelectedFilter]);

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
        <Flex columnGap="0.5rem">
          <ModelFilterButton
            label={t('modelmanager:allModels')}
            onClick={() => setIsSelectedFilter('all')}
            isActive={isSelectedFilter === 'all'}
          />
          <ModelFilterButton
            label={t('modelmanager:checkpointModels')}
            onClick={() => setIsSelectedFilter('ckpt')}
            isActive={isSelectedFilter === 'ckpt'}
          />
          <ModelFilterButton
            label={t('modelmanager:diffusersModels')}
            onClick={() => setIsSelectedFilter('diffusers')}
            isActive={isSelectedFilter === 'diffusers'}
          />
        </Flex>
        {renderModelListItems}
      </Flex>
    </Flex>
  );
};

export default ModelList;
