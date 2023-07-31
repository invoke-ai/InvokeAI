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
  useGetLoRAModelsQuery,
  LoRAModelConfigEntity,
} from 'services/api/endpoints/models';
import ModelListItem from './ModelListItem';
import { ALL_BASE_MODELS } from 'services/api/constants';

type ModelListProps = {
  selectedModelId: string | undefined;
  setSelectedModelId: (name: string | undefined) => void;
};

type ModelFormat = 'images' | 'checkpoint' | 'diffusers';

type ModelType = 'main' | 'lora';

type CombinedModelFormat = ModelFormat | 'lora';

const ModelList = (props: ModelListProps) => {
  const { selectedModelId, setSelectedModelId } = props;
  const { t } = useTranslation();
  const [nameFilter, setNameFilter] = useState<string>('');
  const [modelFormatFilter, setModelFormatFilter] =
    useState<CombinedModelFormat>('images');

  const { filteredDiffusersModels } = useGetMainModelsQuery(ALL_BASE_MODELS, {
    selectFromResult: ({ data }) => ({
      filteredDiffusersModels: modelsFilter(
        data,
        'main',
        'diffusers',
        nameFilter
      ),
    }),
  });

  const { filteredCheckpointModels } = useGetMainModelsQuery(ALL_BASE_MODELS, {
    selectFromResult: ({ data }) => ({
      filteredCheckpointModels: modelsFilter(
        data,
        'main',
        'checkpoint',
        nameFilter
      ),
    }),
  });

  const { filteredLoraModels } = useGetLoRAModelsQuery(undefined, {
    selectFromResult: ({ data }) => ({
      filteredLoraModels: modelsFilter(data, 'lora', undefined, nameFilter),
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
          <IAIButton
            size="sm"
            onClick={() => setModelFormatFilter('lora')}
            isChecked={modelFormatFilter === 'lora'}
          >
            {t('modelManager.loraModels')}
          </IAIButton>
        </ButtonGroup>

        <IAIInput
          onChange={handleSearchFilter}
          label={t('modelManager.search')}
          labelPos="side"
        />

        <Flex
          flexDirection="column"
          gap={4}
          maxHeight={window.innerHeight - 280}
          overflow="scroll"
        >
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
                    Checkpoints
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
          {['images', 'lora'].includes(modelFormatFilter) &&
            filteredLoraModels.length > 0 && (
              <StyledModelContainer>
                <Flex sx={{ gap: 2, flexDir: 'column' }}>
                  <Text variant="subtext" fontSize="sm">
                    LoRAs
                  </Text>
                  {filteredLoraModels.map((model) => (
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
    </Flex>
  );
};

export default ModelList;

const modelsFilter = <T extends MainModelConfigEntity | LoRAModelConfigEntity>(
  data: EntityState<T> | undefined,
  model_type: ModelType,
  model_format: ModelFormat | undefined,
  nameFilter: string
) => {
  const filteredModels: T[] = [];
  forEach(data?.entities, (model) => {
    if (!model) {
      return;
    }

    const matchesFilter = model.model_name
      .toLowerCase()
      .includes(nameFilter.toLowerCase());

    const matchesFormat =
      model_format === undefined || model.model_format === model_format;
    const matchesType = model.model_type === model_type;

    if (matchesFilter && matchesFormat && matchesType) {
      filteredModels.push(model);
    }
  });
  return filteredModels;
};

const StyledModelContainer = (props: PropsWithChildren) => {
  return (
    <Flex
      flexDirection="column"
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
