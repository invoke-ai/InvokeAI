import { Flex, Text } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import type { FilterableModelType } from 'features/modelManagerV2/store/modelManagerV2Slice';
import { memo, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import {
  useControlNetModels,
  useEmbeddingModels,
  useIPAdapterModels,
  useLoRAModels,
  useMainModels,
  useRefinerModels,
  useT2IAdapterModels,
  useVAEModels,
} from 'services/api/hooks/modelsByType';
import type { AnyModelConfig } from 'services/api/types';

import { FetchingModelsLoader } from './FetchingModelsLoader';
import { ModelListWrapper } from './ModelListWrapper';

const ModelList = () => {
  const { searchTerm, filteredModelType } = useAppSelector((s) => s.modelmanagerV2);
  const { t } = useTranslation();

  const [mainModels, { isLoading: isLoadingMainModels }] = useMainModels();
  const filteredMainModels = useMemo(
    () => modelsFilter(mainModels, searchTerm, filteredModelType),
    [mainModels, searchTerm, filteredModelType]
  );

  const [refinerModels, { isLoading: isLoadingRefinerModels }] = useRefinerModels();
  const filteredRefinerModels = useMemo(
    () => modelsFilter(refinerModels, searchTerm, filteredModelType),
    [refinerModels, searchTerm, filteredModelType]
  );

  const [loraModels, { isLoading: isLoadingLoRAModels }] = useLoRAModels();
  const filteredLoRAModels = useMemo(
    () => modelsFilter(loraModels, searchTerm, filteredModelType),
    [loraModels, searchTerm, filteredModelType]
  );

  const [embeddingModels, { isLoading: isLoadingEmbeddingModels }] = useEmbeddingModels();
  const filteredEmbeddingModels = useMemo(
    () => modelsFilter(embeddingModels, searchTerm, filteredModelType),
    [embeddingModels, searchTerm, filteredModelType]
  );

  const [controlNetModels, { isLoading: isLoadingControlNetModels }] = useControlNetModels();
  const filteredControlNetModels = useMemo(
    () => modelsFilter(controlNetModels, searchTerm, filteredModelType),
    [controlNetModels, searchTerm, filteredModelType]
  );

  const [t2iAdapterModels, { isLoading: isLoadingT2IAdapterModels }] = useT2IAdapterModels();
  const filteredT2IAdapterModels = useMemo(
    () => modelsFilter(t2iAdapterModels, searchTerm, filteredModelType),
    [t2iAdapterModels, searchTerm, filteredModelType]
  );

  const [ipAdapterModels, { isLoading: isLoadingIPAdapterModels }] = useIPAdapterModels();
  const filteredIPAdapterModels = useMemo(
    () => modelsFilter(ipAdapterModels, searchTerm, filteredModelType),
    [ipAdapterModels, searchTerm, filteredModelType]
  );

  const [vaeModels, { isLoading: isLoadingVAEModels }] = useVAEModels();
  const filteredVAEModels = useMemo(
    () => modelsFilter(vaeModels, searchTerm, filteredModelType),
    [vaeModels, searchTerm, filteredModelType]
  );

  const totalFilteredModels = useMemo(() => {
    return (
      filteredMainModels.length +
      filteredRefinerModels.length +
      filteredLoRAModels.length +
      filteredEmbeddingModels.length +
      filteredControlNetModels.length +
      filteredT2IAdapterModels.length +
      filteredIPAdapterModels.length +
      filteredVAEModels.length
    );
  }, [
    filteredControlNetModels.length,
    filteredEmbeddingModels.length,
    filteredIPAdapterModels.length,
    filteredLoRAModels.length,
    filteredMainModels.length,
    filteredRefinerModels.length,
    filteredT2IAdapterModels.length,
    filteredVAEModels.length,
  ]);

  return (
    <ScrollableContent>
      <Flex flexDirection="column" w="full" h="full" gap={4}>
        {/* Main Model List */}
        {isLoadingMainModels && <FetchingModelsLoader loadingMessage="Loading Main Models..." />}
        {!isLoadingMainModels && filteredMainModels.length > 0 && (
          <ModelListWrapper title={t('modelManager.main')} modelList={filteredMainModels} key="main" />
        )}
        {/* Refiner Model List */}
        {isLoadingRefinerModels && <FetchingModelsLoader loadingMessage="Loading Refiner Models..." />}
        {!isLoadingRefinerModels && filteredRefinerModels.length > 0 && (
          <ModelListWrapper title={t('sdxl.refiner')} modelList={filteredRefinerModels} key="refiner" />
        )}
        {/* LoRAs List */}
        {isLoadingLoRAModels && <FetchingModelsLoader loadingMessage="Loading LoRAs..." />}
        {!isLoadingLoRAModels && filteredLoRAModels.length > 0 && (
          <ModelListWrapper title={t('modelManager.loraModels')} modelList={filteredLoRAModels} key="loras" />
        )}

        {/* TI List */}
        {isLoadingEmbeddingModels && <FetchingModelsLoader loadingMessage="Loading Textual Inversions..." />}
        {!isLoadingEmbeddingModels && filteredEmbeddingModels.length > 0 && (
          <ModelListWrapper
            title={t('modelManager.textualInversions')}
            modelList={filteredEmbeddingModels}
            key="textual-inversions"
          />
        )}

        {/* VAE List */}
        {isLoadingVAEModels && <FetchingModelsLoader loadingMessage="Loading VAEs..." />}
        {!isLoadingVAEModels && filteredVAEModels.length > 0 && (
          <ModelListWrapper title="VAE" modelList={filteredVAEModels} key="vae" />
        )}

        {/* Controlnet List */}
        {isLoadingControlNetModels && <FetchingModelsLoader loadingMessage="Loading ControlNets..." />}
        {!isLoadingControlNetModels && filteredControlNetModels.length > 0 && (
          <ModelListWrapper title="ControlNet" modelList={filteredControlNetModels} key="controlnets" />
        )}
        {/* IP Adapter List */}
        {isLoadingIPAdapterModels && <FetchingModelsLoader loadingMessage="Loading IP Adapters..." />}
        {!isLoadingIPAdapterModels && filteredIPAdapterModels.length > 0 && (
          <ModelListWrapper title={t('common.ipAdapter')} modelList={filteredIPAdapterModels} key="ip-adapters" />
        )}
        {/* T2I Adapters List */}
        {isLoadingT2IAdapterModels && <FetchingModelsLoader loadingMessage="Loading T2I Adapters..." />}
        {!isLoadingT2IAdapterModels && filteredT2IAdapterModels.length > 0 && (
          <ModelListWrapper title={t('common.t2iAdapter')} modelList={filteredT2IAdapterModels} key="t2i-adapters" />
        )}
        {totalFilteredModels === 0 && (
          <Flex w="full" h="full" alignItems="center" justifyContent="center">
            <Text>{t('modelManager.noMatchingModels')}</Text>
          </Flex>
        )}
      </Flex>
    </ScrollableContent>
  );
};

export default memo(ModelList);

const modelsFilter = <T extends AnyModelConfig>(
  data: T[],
  nameFilter: string,
  filteredModelType: FilterableModelType | null
): T[] => {
  return data.filter((model) => {
    const matchesFilter = model.name.toLowerCase().includes(nameFilter.toLowerCase());
    const matchesType = getMatchesType(model, filteredModelType);

    return matchesFilter && matchesType;
  });
};

const getMatchesType = (modelConfig: AnyModelConfig, filteredModelType: FilterableModelType | null): boolean => {
  if (filteredModelType === 'refiner') {
    return modelConfig.base === 'sdxl-refiner';
  }

  if (filteredModelType === 'main' && modelConfig.base === 'sdxl-refiner') {
    return false;
  }

  return filteredModelType ? modelConfig.type === filteredModelType : true;
};
