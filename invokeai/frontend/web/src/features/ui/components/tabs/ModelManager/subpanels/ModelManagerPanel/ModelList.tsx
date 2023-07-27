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
  OnnxModelConfigEntity,
  useGetMainModelsQuery,
  useGetOnnxModelsQuery,
} from 'services/api/endpoints/models';
import ModelListItem from './ModelListItem';
import { ALL_BASE_MODELS } from 'services/api/constants';

type ModelListProps = {
  selectedModelId: string | undefined;
  setSelectedModelId: (name: string | undefined) => void;
};

type ModelFormat = 'images' | 'checkpoint' | 'diffusers' | 'olive' | 'onnx';

const ModelList = (props: ModelListProps) => {
  const { selectedModelId, setSelectedModelId } = props;
  const { t } = useTranslation();
  const [nameFilter, setNameFilter] = useState<string>('');
  const [modelFormatFilter, setModelFormatFilter] =
    useState<ModelFormat>('images');

  const { filteredDiffusersModels } = useGetMainModelsQuery(ALL_BASE_MODELS, {
    selectFromResult: ({ data }) => ({
      filteredDiffusersModels: modelsFilter(data, 'diffusers', nameFilter),
    }),
  });

  const { filteredCheckpointModels } = useGetMainModelsQuery(ALL_BASE_MODELS, {
    selectFromResult: ({ data }) => ({
      filteredCheckpointModels: modelsFilter(data, 'checkpoint', nameFilter),
    }),
  });

  const { filteredOnnxModels } = useGetOnnxModelsQuery(ALL_BASE_MODELS, {
    selectFromResult: ({ data }) => ({
      filteredOnnxModels: modelsFilter(data, 'onnx', nameFilter),
    }),
  });

  const { filteredOliveModels } = useGetOnnxModelsQuery(ALL_BASE_MODELS, {
    selectFromResult: ({ data }) => ({
      filteredOliveModels: modelsFilter(data, 'olive', nameFilter),
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
            onClick={() => setModelFormatFilter('onnx')}
            isChecked={modelFormatFilter === 'onnx'}
          >
            {t('modelManager.onnxModels')}
          </IAIButton>
          <IAIButton
            size="sm"
            onClick={() => setModelFormatFilter('olive')}
            isChecked={modelFormatFilter === 'olive'}
          >
            {t('modelManager.oliveModels')}
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
          {['images', 'olive'].includes(modelFormatFilter) &&
            filteredOliveModels.length > 0 && (
              <StyledModelContainer>
                <Flex sx={{ gap: 2, flexDir: 'column' }}>
                  <Text variant="subtext" fontSize="sm">
                    Olives
                  </Text>
                  {filteredOliveModels.map((model) => (
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
          {['images', 'onnx'].includes(modelFormatFilter) &&
            filteredOnnxModels.length > 0 && (
              <StyledModelContainer>
                <Flex sx={{ gap: 2, flexDir: 'column' }}>
                  <Text variant="subtext" fontSize="sm">
                    Onnx
                  </Text>
                  {filteredOnnxModels.map((model) => (
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

const modelsFilter = (
  data:
    | EntityState<MainModelConfigEntity>
    | EntityState<OnnxModelConfigEntity>
    | undefined,
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
