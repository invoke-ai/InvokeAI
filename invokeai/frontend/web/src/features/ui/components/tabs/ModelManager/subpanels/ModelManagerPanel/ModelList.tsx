import { ButtonGroup, Flex, Text } from '@chakra-ui/react';
import { EntityState } from '@reduxjs/toolkit';
import IAIButton from 'common/components/IAIButton';
import IAIInput from 'common/components/IAIInput';
import { forEach } from 'lodash-es';
import type { ChangeEvent, PropsWithChildren } from 'react';
import { useCallback, useState } from 'react';
import { useTranslation } from 'react-i18next';
import {
  MainModelConfigEntity,
  useGetMainModelsQuery,
} from 'services/api/endpoints/models';
import ModelListItem from './ModelListItem';

type ModelListProps = {
  selectedModelId: string | undefined;
  setSelectedModelId: (name: string | undefined) => void;
};

type ModelFormat = 'images' | 'checkpoint' | 'diffusers';

const ModelList = (props: ModelListProps) => {
  const { selectedModelId, setSelectedModelId } = props;
  const { t } = useTranslation();
  const [nameFilter, setNameFilter] = useState<string>('');
  const [modelFormatFilter, setModelFormatFilter] =
    useState<ModelFormat>('images');

  const { filteredDiffusersModels } = useGetMainModelsQuery(undefined, {
    selectFromResult: ({ data }) => ({
      filteredDiffusersModels: modelsFilter(data, 'diffusers', nameFilter),
    }),
  });

  const { filteredCheckpointModels } = useGetMainModelsQuery(undefined, {
    selectFromResult: ({ data }) => ({
      filteredCheckpointModels: modelsFilter(data, 'checkpoint', nameFilter),
    }),
  });

  const handleSearchFilter = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    setNameFilter(e.target.value);
  }, []);

  return (
    <Flex flexDirection="column" rowGap={4} width="50%" minWidth="50%">
      <Flex flexDirection="column" gap={4} paddingInlineEnd={4}>
        <ButtonGroup isAttached>
          <IAIButton
            onClick={() => setModelFormatFilter('images')}
            isChecked={modelFormatFilter === 'images'}
            size="sm"
          >
            {t('modelManager.allModels')}
          </IAIButton>
          <IAIButton
            size="sm"
            onClick={() => setModelFormatFilter('diffusers')}
            isChecked={modelFormatFilter === 'diffusers'}
          >
            {t('modelManager.diffusersModels')}
          </IAIButton>
          <IAIButton
            size="sm"
            onClick={() => setModelFormatFilter('checkpoint')}
            isChecked={modelFormatFilter === 'checkpoint'}
          >
            {t('modelManager.checkpointModels')}
          </IAIButton>
        </ButtonGroup>

        <IAIInput
          onChange={handleSearchFilter}
          label={t('modelManager.search')}
          labelPos="side"
        />

        {['images', 'diffusers'].includes(modelFormatFilter) &&
          filteredDiffusersModels.length > 0 && (
            <StyledModelContainer>
              <Flex sx={{ gap: 2, flexDir: 'column' }}>
                <Text variant="subtext" fontSize="sm">
                  Diffusers
                </Text>
                {filteredDiffusersModels.map((model) => (
                  <ModelListItem
                    key={model.id}
                    model={model}
                    isSelected={selectedModelId === model.id}
                    setSelectedModelId={setSelectedModelId}
                  />
                ))}
              </Flex>
            </StyledModelContainer>
          )}
        {['images', 'checkpoint'].includes(modelFormatFilter) &&
          filteredCheckpointModels.length > 0 && (
            <StyledModelContainer>
              <Flex sx={{ gap: 2, flexDir: 'column' }}>
                <Text variant="subtext" fontSize="sm">
                  Checkpoint
                </Text>
                {filteredCheckpointModels.map((model) => (
                  <ModelListItem
                    key={model.id}
                    model={model}
                    isSelected={selectedModelId === model.id}
                    setSelectedModelId={setSelectedModelId}
                  />
                ))}
              </Flex>
            </StyledModelContainer>
          )}
      </Flex>
    </Flex>
  );
};

export default ModelList;

const modelsFilter = (
  data: EntityState<MainModelConfigEntity> | undefined,
  model_format: ModelFormat,
  nameFilter: string
) => {
  const filteredModels: MainModelConfigEntity[] = [];
  forEach(data?.entities, (model) => {
    if (!model) {
      return;
    }

    const matchesFilter = model.model_name
      .toLowerCase()
      .includes(nameFilter.toLowerCase());

    const matchesFormat = model.model_format === model_format;

    if (matchesFilter && matchesFormat) {
      filteredModels.push(model);
    }
  });
  return filteredModels;
};

const StyledModelContainer = (props: PropsWithChildren) => {
  return (
    <Flex
      flexDirection="column"
      maxHeight={window.innerHeight - 280}
      overflow="scroll"
      gap={4}
      borderRadius={4}
      p={4}
      sx={{
        bg: 'base.200',
        _dark: {
          bg: 'base.800',
        },
      }}
    >
      {props.children}
    </Flex>
  );
};
