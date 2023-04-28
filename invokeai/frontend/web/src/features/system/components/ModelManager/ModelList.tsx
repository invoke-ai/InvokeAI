import { Box, Flex, Heading, Spacer, Spinner, Text } from '@chakra-ui/react';
import IAIInput from 'common/components/IAIInput';
import IAIButton from 'common/components/IAIButton';

import AddModel from './AddModel';
import ModelListItem from './ModelListItem';
import MergeModels from './MergeModels';

import { useAppSelector } from 'app/store/storeHooks';
import { useTranslation } from 'react-i18next';

import { createSelector } from '@reduxjs/toolkit';
import { systemSelector } from 'features/system/store/systemSelectors';
import type { SystemState } from 'features/system/store/systemSlice';
import { isEqual, map } from 'lodash-es';

import React, { useMemo, useState, useTransition } from 'react';
import type { ChangeEvent, ReactNode } from 'react';

const modelListSelector = createSelector(
  systemSelector,
  (system: SystemState) => {
    const models = map(system.model_list, (model, key) => {
      return { name: key, ...model };
    });
    return models;
  },
  {
    memoizeOptions: {
      resultEqualityCheck: isEqual,
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
      sx={{
        _active: {
          bg: 'accent.750',
        },
      }}
      size="sm"
    >
      {label}
    </IAIButton>
  );
}

const ModelList = () => {
  const models = useAppSelector(modelListSelector);

  const [renderModelList, setRenderModelList] = React.useState<boolean>(false);

  React.useEffect(() => {
    const timer = setTimeout(() => {
      setRenderModelList(true);
    }, 200);

    return () => clearTimeout(timer);
  }, []);

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
        <Box marginTop={4}>{filteredModelListItemsToRender}</Box>
      ) : (
        <Box marginTop={4}>{localFilteredModelListItemsToRender}</Box>
      )
    ) : (
      <Flex flexDirection="column" rowGap={6}>
        {isSelectedFilter === 'all' && (
          <>
            <Box>
              <Text
                sx={{
                  fontWeight: '500',
                  py: 2,
                  px: 4,
                  my: 4,
                  mx: 0,
                  borderRadius: 'base',
                  width: 'max-content',
                  fontSize: 'sm',
                  bg: 'base.750',
                }}
              >
                {t('modelManager.checkpointModels')}
              </Text>
              {ckptModelListItemsToRender}
            </Box>
            <Box>
              <Text
                sx={{
                  fontWeight: '500',
                  py: 2,
                  px: 4,
                  mb: 4,
                  borderRadius: 'base',
                  width: 'max-content',
                  fontSize: 'sm',
                  bg: 'base.750',
                }}
              >
                {t('modelManager.diffusersModels')}
              </Text>
              {diffusersModelListItemsToRender}
            </Box>
          </>
        )}

        {isSelectedFilter === 'ckpt' && (
          <Flex flexDirection="column" marginTop={4}>
            {ckptModelListItemsToRender}
          </Flex>
        )}

        {isSelectedFilter === 'diffusers' && (
          <Flex flexDirection="column" marginTop={4}>
            {diffusersModelListItemsToRender}
          </Flex>
        )}
      </Flex>
    );
  }, [models, searchText, t, isSelectedFilter]);

  return (
    <Flex flexDirection="column" rowGap={4} width="50%" minWidth="50%">
      <Flex justifyContent="space-between" alignItems="center" gap={2}>
        <Heading size="md">{t('modelManager.availableModels')}</Heading>
        <Spacer />
        <AddModel />
        <MergeModels />
      </Flex>

      <IAIInput
        onChange={handleSearchFilter}
        label={t('modelManager.search')}
      />

      <Flex
        flexDirection="column"
        gap={1}
        maxHeight={window.innerHeight - 240}
        overflow="scroll"
        paddingInlineEnd={4}
      >
        <Flex columnGap={2}>
          <ModelFilterButton
            label={t('modelManager.allModels')}
            onClick={() => setIsSelectedFilter('all')}
            isActive={isSelectedFilter === 'all'}
          />
          <ModelFilterButton
            label={t('modelManager.checkpointModels')}
            onClick={() => setIsSelectedFilter('ckpt')}
            isActive={isSelectedFilter === 'ckpt'}
          />
          <ModelFilterButton
            label={t('modelManager.diffusersModels')}
            onClick={() => setIsSelectedFilter('diffusers')}
            isActive={isSelectedFilter === 'diffusers'}
          />
        </Flex>

        {renderModelList ? (
          renderModelListItems
        ) : (
          <Flex
            width="100%"
            minHeight={96}
            justifyContent="center"
            alignItems="center"
          >
            <Spinner />
          </Flex>
        )}
      </Flex>
    </Flex>
  );
};

export default ModelList;
