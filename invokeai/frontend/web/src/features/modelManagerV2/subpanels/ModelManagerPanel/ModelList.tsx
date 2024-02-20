import { Flex, Spinner, Text } from '@invoke-ai/ui-library';
import type { EntityState } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { forEach } from 'lodash-es';
import { memo } from 'react';
import { ALL_BASE_MODELS } from 'services/api/constants';
import {
  useGetControlNetModelsQuery,
  useGetIPAdapterModelsQuery,
  useGetLoRAModelsQuery,
  useGetMainModelsQuery,
  useGetT2IAdapterModelsQuery,
  useGetTextualInversionModelsQuery,
  useGetVaeModelsQuery,
} from 'services/api/endpoints/models';
import type { AnyModelConfig } from 'services/api/types';

import { ModelListWrapper } from './ModelListWrapper';

const ModelList = () => {
  const { searchTerm, filteredModelType } = useAppSelector((s) => s.modelmanagerV2);

  const { filteredMainModels, isLoadingMainModels } = useGetMainModelsQuery(ALL_BASE_MODELS, {
    selectFromResult: ({ data, isLoading }) => ({
      filteredMainModels: modelsFilter(data, searchTerm, filteredModelType),
      isLoadingMainModels: isLoading,
    }),
  });

  const { filteredLoraModels, isLoadingLoraModels } = useGetLoRAModelsQuery(undefined, {
    selectFromResult: ({ data, isLoading }) => ({
      filteredLoraModels: modelsFilter(data, searchTerm, filteredModelType),
      isLoadingLoraModels: isLoading,
    }),
  });

  const { filteredTextualInversionModels, isLoadingTextualInversionModels } = useGetTextualInversionModelsQuery(
    undefined,
    {
      selectFromResult: ({ data, isLoading }) => ({
        filteredTextualInversionModels: modelsFilter(data, searchTerm, filteredModelType),
        isLoadingTextualInversionModels: isLoading,
      }),
    }
  );

  const { filteredControlnetModels, isLoadingControlnetModels } = useGetControlNetModelsQuery(undefined, {
    selectFromResult: ({ data, isLoading }) => ({
      filteredControlnetModels: modelsFilter(data, searchTerm, filteredModelType),
      isLoadingControlnetModels: isLoading,
    }),
  });

  const { filteredT2iAdapterModels, isLoadingT2IAdapterModels } = useGetT2IAdapterModelsQuery(undefined, {
    selectFromResult: ({ data, isLoading }) => ({
      filteredT2iAdapterModels: modelsFilter(data, searchTerm, filteredModelType),
      isLoadingT2IAdapterModels: isLoading,
    }),
  });

  const { filteredIpAdapterModels, isLoadingIpAdapterModels } = useGetIPAdapterModelsQuery(undefined, {
    selectFromResult: ({ data, isLoading }) => ({
      filteredIpAdapterModels: modelsFilter(data, searchTerm, filteredModelType),
      isLoadingIpAdapterModels: isLoading,
    }),
  });

  const { filteredVaeModels, isLoadingVaeModels } = useGetVaeModelsQuery(undefined, {
    selectFromResult: ({ data, isLoading }) => ({
      filteredVaeModels: modelsFilter(data, searchTerm, filteredModelType),
      isLoadingVaeModels: isLoading,
    }),
  });

  return (
    <Flex flexDirection="column" p={4}>
      <Flex flexDirection="column" maxHeight={window.innerHeight - 130} overflow="scroll">
        {/* Main Model List */}
        {isLoadingMainModels && <FetchingModelsLoader loadingMessage="Loading Main..." />}
        {!isLoadingMainModels && filteredMainModels.length > 0 && (
          <ModelListWrapper title="Main" modelList={filteredMainModels} key="main" />
        )}
        {/* LoRAs List */}
        {isLoadingLoraModels && <FetchingModelsLoader loadingMessage="Loading LoRAs..." />}
        {!isLoadingLoraModels && filteredLoraModels.length > 0 && (
          <ModelListWrapper title="LoRAs" modelList={filteredLoraModels} key="loras" />
        )}

        {/* TI List */}
        {isLoadingTextualInversionModels && <FetchingModelsLoader loadingMessage="Loading Textual Inversions..." />}
        {!isLoadingTextualInversionModels && filteredTextualInversionModels.length > 0 && (
          <ModelListWrapper
            title="Textual Inversions"
            modelList={filteredTextualInversionModels}
            key="textual-inversions"
          />
        )}

        {/* VAE List */}
        {isLoadingVaeModels && <FetchingModelsLoader loadingMessage="Loading VAEs..." />}
        {!isLoadingVaeModels && filteredVaeModels.length > 0 && (
          <ModelListWrapper title="VAEs" modelList={filteredVaeModels} key="vae" />
        )}

        {/* Controlnet List */}
        {isLoadingControlnetModels && <FetchingModelsLoader loadingMessage="Loading Controlnets..." />}
        {!isLoadingControlnetModels && filteredControlnetModels.length > 0 && (
          <ModelListWrapper title="Controlnets" modelList={filteredControlnetModels} key="controlnets" />
        )}
        {/* IP Adapter List */}
        {isLoadingIpAdapterModels && <FetchingModelsLoader loadingMessage="Loading IP Adapters..." />}
        {!isLoadingIpAdapterModels && filteredIpAdapterModels.length > 0 && (
          <ModelListWrapper title="IP Adapters" modelList={filteredIpAdapterModels} key="ip-adapters" />
        )}
        {/* T2I Adapters List */}
        {isLoadingT2IAdapterModels && <FetchingModelsLoader loadingMessage="Loading T2I Adapters..." />}
        {!isLoadingT2IAdapterModels && filteredT2iAdapterModels.length > 0 && (
          <ModelListWrapper title="T2I Adapters" modelList={filteredT2iAdapterModels} key="t2i-adapters" />
        )}
      </Flex>
    </Flex>
  );
};

export default memo(ModelList);

const modelsFilter = <T extends AnyModelConfig>(
  data: EntityState<T, string> | undefined,
  nameFilter: string,
  filteredModelType: string | null
): T[] => {
  const filteredModels: T[] = [];

  forEach(data?.entities, (model) => {
    if (!model) {
      return;
    }

    const matchesFilter = model.name.toLowerCase().includes(nameFilter.toLowerCase());
    const matchesType = filteredModelType ? model.type === filteredModelType : true;

    if (matchesFilter && matchesType) {
      filteredModels.push(model);
    }
  });
  return filteredModels;
};

const FetchingModelsLoader = memo(({ loadingMessage }: { loadingMessage?: string }) => {
  return (
    <Flex flexDirection="column" gap={4} borderRadius={4} p={4} bg="base.800">
      <Flex justifyContent="center" alignItems="center" flexDirection="column" p={4} gap={8}>
        <Spinner />
        <Text variant="subtext">{loadingMessage ? loadingMessage : 'Fetching...'}</Text>
      </Flex>
    </Flex>
  );
});

FetchingModelsLoader.displayName = 'FetchingModelsLoader';
