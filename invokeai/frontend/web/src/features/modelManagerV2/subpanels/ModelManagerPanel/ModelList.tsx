import { Flex, Text, useToast } from '@invoke-ai/ui-library';
import { logger } from 'app/logging/logger';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import { buildUseDisclosure } from 'common/hooks/useBoolean';
import { MODEL_CATEGORIES_AS_LIST } from 'features/modelManagerV2/models';
import {
  clearModelSelection,
  type FilterableModelType,
  selectFilteredModelType,
  selectSearchTerm,
  selectSelectedModelKeys,
  setSelectedModelKey,
} from 'features/modelManagerV2/store/modelManagerV2Slice';
import { memo, useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { serializeError } from 'serialize-error';
import {
  modelConfigsAdapterSelectors,
  useBulkDeleteModelsMutation,
  useGetMissingModelsQuery,
  useGetModelConfigsQuery,
} from 'services/api/endpoints/models';
import type { AnyModelConfig } from 'services/api/types';

import { BulkDeleteModelsModal } from './BulkDeleteModelsModal';
import { FetchingModelsLoader } from './FetchingModelsLoader';
import { MissingModelsProvider } from './MissingModelsContext';
import { ModelListWrapper } from './ModelListWrapper';

const log = logger('models');

export const [useBulkDeleteModal] = buildUseDisclosure(false);

const ModelList = () => {
  const dispatch = useAppDispatch();
  const filteredModelType = useAppSelector(selectFilteredModelType);
  const searchTerm = useAppSelector(selectSearchTerm);
  const selectedModelKeys = useAppSelector(selectSelectedModelKeys);
  const { t } = useTranslation();
  const toast = useToast();
  const { isOpen, close } = useBulkDeleteModal();
  const [isDeleting, setIsDeleting] = useState(false);

  const { data: allModelsData, isLoading: isLoadingAll } = useGetModelConfigsQuery();
  const { data: missingModelsData, isLoading: isLoadingMissing } = useGetMissingModelsQuery();
  const [bulkDeleteModels] = useBulkDeleteModelsMutation();

  const data = filteredModelType === 'missing' ? missingModelsData : allModelsData;
  const isLoading = filteredModelType === 'missing' ? isLoadingMissing : isLoadingAll;

  const models = useMemo(() => {
    const modelConfigs = modelConfigsAdapterSelectors.selectAll(data ?? { ids: [], entities: {} });

    // For missing models filter, show all models in a single category
    if (filteredModelType === 'missing') {
      const filtered = modelConfigs.filter(
        (m) =>
          m.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
          m.base.toLowerCase().includes(searchTerm.toLowerCase()) ||
          m.type.toLowerCase().includes(searchTerm.toLowerCase())
      );
      return {
        total: filtered.length,
        byCategory: [{ i18nKey: 'modelManager.missingFiles', configs: filtered }],
      };
    }

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

  const handleConfirmBulkDelete = useCallback(async () => {
    setIsDeleting(true);
    try {
      const result = await bulkDeleteModels({ keys: selectedModelKeys }).unwrap();

      // Clear selection and close modal
      dispatch(clearModelSelection());
      dispatch(setSelectedModelKey(null));
      close();

      // Show success/failure toast
      if (result.failed.length === 0) {
        toast({
          id: 'BULK_DELETE_SUCCESS',
          title: t('modelManager.modelsDeleted', {
            count: result.deleted.length,
            defaultValue: `Successfully deleted ${result.deleted.length} model(s)`,
          }),
          status: 'success',
        });
      } else if (result.deleted.length === 0) {
        toast({
          id: 'BULK_DELETE_FAILED',
          title: t('modelManager.modelsDeleteFailed', {
            defaultValue: 'Failed to delete models',
          }),
          description: t('modelManager.someModelsFailedToDelete', {
            count: result.failed.length,
            defaultValue: `${result.failed.length} model(s) could not be deleted`,
          }),
          status: 'error',
        });
      } else {
        // Partial success
        toast({
          id: 'BULK_DELETE_PARTIAL',
          title: t('modelManager.modelsDeletedPartial', {
            defaultValue: 'Partially completed',
          }),
          description: t('modelManager.someModelsDeleted', {
            deleted: result.deleted.length,
            failed: result.failed.length,
            defaultValue: `${result.deleted.length} deleted, ${result.failed.length} failed`,
          }),
          status: 'warning',
        });
      }

      log.info(`Bulk delete completed: ${result.deleted.length} deleted, ${result.failed.length} failed`);
    } catch (err) {
      log.error({ error: serializeError(err as Error) }, 'Bulk delete error');
      toast({
        id: 'BULK_DELETE_ERROR',
        title: t('modelManager.modelsDeleteError', {
          defaultValue: 'Error deleting models',
        }),
        status: 'error',
      });
    } finally {
      setIsDeleting(false);
    }
  }, [bulkDeleteModels, selectedModelKeys, dispatch, close, toast, t]);

  return (
    <MissingModelsProvider>
      <Flex flexDirection="column" w="full" h="full">
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
      </Flex>

      <BulkDeleteModelsModal
        isOpen={isOpen}
        onClose={close}
        onConfirm={handleConfirmBulkDelete}
        modelCount={selectedModelKeys.length}
        isDeleting={isDeleting}
      />
    </MissingModelsProvider>
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
