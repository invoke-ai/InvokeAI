import { EMPTY_ARRAY } from 'app/store/constants';
import { useCallback } from 'react';
import { modelConfigsAdapterSelectors, useGetModelConfigsQuery } from 'services/api/endpoints/models';
import type { StarterModel } from 'services/api/types';

type ModelInstallArg = {
  config: Pick<StarterModel, 'name' | 'base' | 'type' | 'description' | 'format'>;
  source: string;
};

/**
 * Flattens a starter model and its dependencies into a list of models, including the starter model itself.
 */
export const flattenStarterModel = (starterModel: StarterModel): StarterModel[] => {
  return [starterModel, ...(starterModel.dependencies || [])];
};

export const useBuildModelInstallArg = () => {
  const { modelList } = useGetModelConfigsQuery(undefined, {
    selectFromResult: ({ data }) => ({ modelList: data ? modelConfigsAdapterSelectors.selectAll(data) : EMPTY_ARRAY }),
  });

  const getIsInstalled = useCallback(
    ({ source, name, base, type, is_installed }: StarterModel): boolean =>
      modelList.some(
        (mc) => is_installed || source === mc.source || (base === mc.base && name === mc.name && type === mc.type)
      ),
    [modelList]
  );

  const buildModelInstallArg = useCallback((starterModel: StarterModel): ModelInstallArg => {
    const { name, base, type, source, description, format } = starterModel;

    return {
      config: {
        name,
        base,
        type,
        description,
        format,
      },
      source,
    };
  }, []);

  return { getIsInstalled, buildModelInstallArg };
};
