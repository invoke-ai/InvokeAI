import type { EntityState } from '@reduxjs/toolkit';
import { forEach } from 'lodash-es';
import { useCallback } from 'react';
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

export const useIsImported = () => {
  const { data: mainModels } = useGetMainModelsQuery(ALL_BASE_MODELS);
  const { data: loras } = useGetLoRAModelsQuery();
  const { data: embeddings } = useGetTextualInversionModelsQuery();
  const { data: controlnets } = useGetControlNetModelsQuery();
  const { data: ipAdapters } = useGetIPAdapterModelsQuery();
  const { data: t2is } = useGetT2IAdapterModelsQuery();
  const { data: vaes } = useGetVaeModelsQuery();

  const isImported = useCallback(
    ({ name }: { name: string }) => {
      const data = [mainModels, loras, embeddings, controlnets, ipAdapters, t2is, vaes];
      let isMatch = false;
      for (let index = 0; index < data.length; index++) {
        const modelType: EntityState<AnyModelConfig, string> | undefined = data[index];

        const match = modelsFilter(modelType, name);

        if (match.length) {
          isMatch = true;
          break;
        }
      }
      return isMatch;
    },
    [mainModels, loras, embeddings, controlnets, ipAdapters, t2is, vaes]
  );

  return { isImported };
};

const modelsFilter = <T extends AnyModelConfig>(data: EntityState<T, string> | undefined, nameFilter: string): T[] => {
  const filteredModels: T[] = [];

  forEach(data?.entities, (model) => {
    if (!model) {
      return;
    }

    const matchesFilter = model.path.toLowerCase().includes(nameFilter.toLowerCase());

    if (matchesFilter) {
      filteredModels.push(model);
    }
  });
  return filteredModels;
};
