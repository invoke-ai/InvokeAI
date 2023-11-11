import { ButtonGroup, Flex, Spinner, Text } from '@chakra-ui/react';
import { EntityState } from '@reduxjs/toolkit';
import IAIButton from 'common/components/IAIButton';
import IAIInput from 'common/components/IAIInput';
import { forEach } from 'lodash-es';
import { ChangeEvent, PropsWithChildren, memo } from 'react';
import { useCallback, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { ALL_BASE_MODELS } from 'services/api/constants';
import {
  LoRAModelConfigEntity,
  MainModelConfigEntity,
  OnnxModelConfigEntity,
  useGetLoRAModelsQuery,
  useGetMainModelsQuery,
  useGetOnnxModelsQuery,
} from 'services/api/endpoints/models';
import ModelListItem from './ModelListItem';

type ModelListProps = {
  selectedModelId: string | undefined;
  setSelectedModelId: (name: string | undefined) => void;
};

type ModelFormat = 'all' | 'checkpoint' | 'diffusers' | 'olive' | 'onnx';

type ModelType = 'main' | 'lora' | 'onnx';

type CombinedModelFormat = ModelFormat | 'lora';

const ModelList = (props: ModelListProps) => {
  const { selectedModelId, setSelectedModelId } = props;
  const { t } = useTranslation();
  const [nameFilter, setNameFilter] = useState<string>('');
  const [modelFormatFilter, setModelFormatFilter] =
    useState<CombinedModelFormat>('all');

  const { filteredDiffusersModels, isLoadingDiffusersModels } =
    useGetMainModelsQuery(ALL_BASE_MODELS, {
      selectFromResult: ({ data, isLoading }) => ({
        filteredDiffusersModels: modelsFilter(
          data,
          'main',
          'diffusers',
          nameFilter
        ),
        isLoadingDiffusersModels: isLoading,
      }),
    });

  const { filteredCheckpointModels, isLoadingCheckpointModels } =
    useGetMainModelsQuery(ALL_BASE_MODELS, {
      selectFromResult: ({ data, isLoading }) => ({
        filteredCheckpointModels: modelsFilter(
          data,
          'main',
          'checkpoint',
          nameFilter
        ),
        isLoadingCheckpointModels: isLoading,
      }),
    });

  const { filteredLoraModels, isLoadingLoraModels } = useGetLoRAModelsQuery(
    undefined,
    {
      selectFromResult: ({ data, isLoading }) => ({
        filteredLoraModels: modelsFilter(data, 'lora', undefined, nameFilter),
        isLoadingLoraModels: isLoading,
      }),
    }
  );

  const { filteredOnnxModels, isLoadingOnnxModels } = useGetOnnxModelsQuery(
    ALL_BASE_MODELS,
    {
      selectFromResult: ({ data, isLoading }) => ({
        filteredOnnxModels: modelsFilter(data, 'onnx', 'onnx', nameFilter),
        isLoadingOnnxModels: isLoading,
      }),
    }
  );

  const { filteredOliveModels, isLoadingOliveModels } = useGetOnnxModelsQuery(
    ALL_BASE_MODELS,
    {
      selectFromResult: ({ data, isLoading }) => ({
        filteredOliveModels: modelsFilter(data, 'onnx', 'olive', nameFilter),
        isLoadingOliveModels: isLoading,
      }),
    }
  );

  const handleSearchFilter = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    setNameFilter(e.target.value);
  }, []);

  return (
    <Flex flexDirection="column" rowGap={4} width="50%" minWidth="50%">
      <Flex flexDirection="column" gap={4} paddingInlineEnd={4}>
        <ButtonGroup isAttached>
          <IAIButton
            onClick={() => setModelFormatFilter('all')}
            isChecked={modelFormatFilter === 'all'}
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
          {/* Diffusers List */}
          {isLoadingDiffusersModels && (
            <FetchingModelsLoader loadingMessage="Loading Diffusers..." />
          )}
          {['all', 'diffusers'].includes(modelFormatFilter) &&
            !isLoadingDiffusersModels &&
            filteredDiffusersModels.length > 0 && (
              <ModelListWrapper
                title="Diffusers"
                modelList={filteredDiffusersModels}
                selected={{ selectedModelId, setSelectedModelId }}
                key="diffusers"
              />
            )}
          {/* Checkpoints List */}
          {isLoadingCheckpointModels && (
            <FetchingModelsLoader loadingMessage="Loading Checkpoints..." />
          )}
          {['all', 'checkpoint'].includes(modelFormatFilter) &&
            !isLoadingCheckpointModels &&
            filteredCheckpointModels.length > 0 && (
              <ModelListWrapper
                title="Checkpoints"
                modelList={filteredCheckpointModels}
                selected={{ selectedModelId, setSelectedModelId }}
                key="checkpoints"
              />
            )}

          {/* LoRAs List */}
          {isLoadingLoraModels && (
            <FetchingModelsLoader loadingMessage="Loading LoRAs..." />
          )}
          {['all', 'lora'].includes(modelFormatFilter) &&
            !isLoadingLoraModels &&
            filteredLoraModels.length > 0 && (
              <ModelListWrapper
                title="LoRAs"
                modelList={filteredLoraModels}
                selected={{ selectedModelId, setSelectedModelId }}
                key="loras"
              />
            )}
          {/* Olive List */}
          {isLoadingOliveModels && (
            <FetchingModelsLoader loadingMessage="Loading Olives..." />
          )}
          {['all', 'olive'].includes(modelFormatFilter) &&
            !isLoadingOliveModels &&
            filteredOliveModels.length > 0 && (
              <ModelListWrapper
                title="Olives"
                modelList={filteredOliveModels}
                selected={{ selectedModelId, setSelectedModelId }}
                key="olive"
              />
            )}
          {/* Onnx List */}
          {isLoadingOnnxModels && (
            <FetchingModelsLoader loadingMessage="Loading ONNX..." />
          )}
          {['all', 'onnx'].includes(modelFormatFilter) &&
            !isLoadingOnnxModels &&
            filteredOnnxModels.length > 0 && (
              <ModelListWrapper
                title="ONNX"
                modelList={filteredOnnxModels}
                selected={{ selectedModelId, setSelectedModelId }}
                key="onnx"
              />
            )}
        </Flex>
      </Flex>
    </Flex>
  );
};

export default memo(ModelList);

const modelsFilter = <
  T extends
    | MainModelConfigEntity
    | LoRAModelConfigEntity
    | OnnxModelConfigEntity,
>(
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

const StyledModelContainer = memo((props: PropsWithChildren) => {
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
});

StyledModelContainer.displayName = 'StyledModelContainer';

type ModelListWrapperProps = {
  title: string;
  modelList:
    | MainModelConfigEntity[]
    | LoRAModelConfigEntity[]
    | OnnxModelConfigEntity[];
  selected: ModelListProps;
};

const ModelListWrapper = memo((props: ModelListWrapperProps) => {
  const { title, modelList, selected } = props;
  return (
    <StyledModelContainer>
      <Flex sx={{ gap: 2, flexDir: 'column' }}>
        <Text variant="subtext" fontSize="sm">
          {title}
        </Text>
        {modelList.map((model) => (
          <ModelListItem
            key={model.id}
            model={model}
            isSelected={selected.selectedModelId === model.id}
            setSelectedModelId={selected.setSelectedModelId}
          />
        ))}
      </Flex>
    </StyledModelContainer>
  );
});

ModelListWrapper.displayName = 'ModelListWrapper';

const FetchingModelsLoader = memo(
  ({ loadingMessage }: { loadingMessage?: string }) => {
    return (
      <StyledModelContainer>
        <Flex
          justifyContent="center"
          alignItems="center"
          flexDirection="column"
          p={4}
          gap={8}
        >
          <Spinner />
          <Text variant="subtext">
            {loadingMessage ? loadingMessage : 'Fetching...'}
          </Text>
        </Flex>
      </StyledModelContainer>
    );
  }
);

FetchingModelsLoader.displayName = 'FetchingModelsLoader';
