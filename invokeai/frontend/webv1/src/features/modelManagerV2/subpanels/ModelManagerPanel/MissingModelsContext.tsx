import type { PropsWithChildren } from 'react';
import { createContext, useContext, useMemo } from 'react';
import { modelConfigsAdapterSelectors, useGetMissingModelsQuery } from 'services/api/endpoints/models';

type MissingModelsContextValue = {
  missingModelKeys: Set<string>;
  isLoading: boolean;
};

const MissingModelsContext = createContext<MissingModelsContextValue>({
  missingModelKeys: new Set(),
  isLoading: false,
});

export const MissingModelsProvider = ({ children }: PropsWithChildren) => {
  const { data, isLoading } = useGetMissingModelsQuery();

  const value = useMemo(() => {
    const missingModels = modelConfigsAdapterSelectors.selectAll(data ?? { ids: [], entities: {} });
    const missingModelKeys = new Set(missingModels.map((m) => m.key));
    return { missingModelKeys, isLoading };
  }, [data, isLoading]);

  return <MissingModelsContext.Provider value={value}>{children}</MissingModelsContext.Provider>;
};

const useMissingModels = () => useContext(MissingModelsContext);

export const useIsModelMissing = (modelKey: string) => {
  const { missingModelKeys } = useMissingModels();
  return missingModelKeys.has(modelKey);
};
