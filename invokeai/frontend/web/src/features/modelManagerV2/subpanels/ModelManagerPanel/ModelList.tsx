import { Flex, Text } from '@invoke-ai/ui-library';
import { logger } from 'app/logging/logger';
import { useAppSelector } from 'app/store/storeHooks';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import { MODEL_CATEGORIES_AS_LIST } from 'features/modelManagerV2/models';
import {
  type FilterableModelType,
  selectFilteredModelType,
  selectSearchTerm,
} from 'features/modelManagerV2/store/modelManagerV2Slice';
import { memo, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { modelConfigsAdapterSelectors, useGetModelConfigsQuery } from 'services/api/endpoints/models';
import type { AnyModelConfig } from 'services/api/types';

import { FetchingModelsLoader } from './FetchingModelsLoader';
import { ModelListWrapper } from './ModelListWrapper';

const log = logger('models');

const ModelList = () => {
  const filteredModelType = useAppSelector(selectFilteredModelType);
  const searchTerm = useAppSelector(selectSearchTerm);
  const { t } = useTranslation();

  const { data, isLoading } = useGetModelConfigsQuery();

  const models = useMemo(() => {
    const modelConfigs = modelConfigsAdapterSelectors.selectAll(data ?? { ids: [], entities: {} });
    const baseFilteredModelConfigs = modelsFilter(modelConfigs, searchTerm, filteredModelType);
    const byCategory: { i18nKey: string; configs: AnyModelConfig[] }[] = [];
    const total = baseFilteredModelConfigs.length;
    let renderedTotal = 0;
    for (const { i18nKey, filter } of MODEL_CATEGORIES_AS_LIST) {
      const configs = baseFilteredModelConfigs.filter(filter);
      renderedTotal += configs.length;
      byCategory.push({ i18nKey, configs });
    }
    if (renderedTotal !== total) {
      const ctx = { total, renderedTotal, difference: total - renderedTotal };
      log.warn(
        ctx,
        `ModelList: Not all models were categorized - ensure all possible models are covered in MODEL_CATEGORIES`
      );
    }
    return { total, byCategory };
  }, [data, filteredModelType, searchTerm]);

  return (
    <ScrollableContent>
      <Flex flexDirection="column" w="full" h="full" gap={4}>
        {isLoading && <FetchingModelsLoader loadingMessage="Loading..." />}
        {models.byCategory.map(({ i18nKey, configs }) => (
          <ModelListWrapper key={i18nKey} title={t(i18nKey)} modelList={configs} />
        ))}
        {!isLoading && models.total === 0 && (
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
    const matchesFilter =
      model.name.toLowerCase().includes(nameFilter.toLowerCase()) ||
      model.base.toLowerCase().includes(nameFilter.toLowerCase()) ||
      model.type.toLowerCase().includes(nameFilter.toLowerCase()) ||
      model.description?.toLowerCase().includes(nameFilter.toLowerCase()) ||
      model.format.toLowerCase().includes(nameFilter.toLowerCase());

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
