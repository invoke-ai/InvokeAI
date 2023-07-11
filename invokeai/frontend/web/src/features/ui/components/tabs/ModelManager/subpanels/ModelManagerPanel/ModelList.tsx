import { Box, Flex, Spinner, Text } from '@chakra-ui/react';
import IAIButton from 'common/components/IAIButton';
import IAIInput from 'common/components/IAIInput';

import ModelListItem from './ModelListItem';

import { useTranslation } from 'react-i18next';

import type { ChangeEvent, ReactNode } from 'react';
import React, { useMemo, useState, useTransition } from 'react';
import { useGetMainModelsQuery } from 'services/api/endpoints/models';

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
  const { data: mainModels } = useGetMainModelsQuery();

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

    if (!mainModels) return;

    const modelList = mainModels.entities;

    Object.keys(modelList).forEach((model, i) => {
      if (
        modelList[model].name.toLowerCase().includes(searchText.toLowerCase())
      ) {
        filteredModelListItemsToRender.push(
          <ModelListItem
            key={i}
            modelKey={model}
            name={modelList[model].name}
            description={modelList[model].description}
          />
        );
        if (modelList[model]?.model_format === isSelectedFilter) {
          localFilteredModelListItemsToRender.push(
            <ModelListItem
              key={i}
              modelKey={model}
              name={modelList[model].name}
              description={modelList[model].description}
            />
          );
        }
      }
      if (modelList[model]?.model_format !== 'diffusers') {
        ckptModelListItemsToRender.push(
          <ModelListItem
            key={i}
            modelKey={model}
            name={modelList[model].name}
            description={modelList[model].description}
          />
        );
      } else {
        diffusersModelListItemsToRender.push(
          <ModelListItem
            key={i}
            modelKey={model}
            name={modelList[model].name}
            description={modelList[model].description}
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
          </>
        )}

        {isSelectedFilter === 'diffusers' && (
          <Flex flexDirection="column" marginTop={4}>
            {diffusersModelListItemsToRender}
          </Flex>
        )}

        {isSelectedFilter === 'ckpt' && (
          <Flex flexDirection="column" marginTop={4}>
            {ckptModelListItemsToRender}
          </Flex>
        )}
      </Flex>
    );
  }, [mainModels, searchText, t, isSelectedFilter]);

  return (
    <Flex flexDirection="column" rowGap={4} width="50%" minWidth="50%">
      <IAIInput
        onChange={handleSearchFilter}
        label={t('modelManager.search')}
      />

      <Flex
        flexDirection="column"
        gap={4}
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
            label={t('modelManager.diffusersModels')}
            onClick={() => setIsSelectedFilter('diffusers')}
            isActive={isSelectedFilter === 'diffusers'}
          />
          <ModelFilterButton
            label={t('modelManager.checkpointModels')}
            onClick={() => setIsSelectedFilter('ckpt')}
            isActive={isSelectedFilter === 'ckpt'}
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
