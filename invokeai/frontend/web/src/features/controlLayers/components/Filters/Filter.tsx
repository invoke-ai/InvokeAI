import { Button, ButtonGroup, Flex, Heading } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { FilterSettings } from 'features/controlLayers/components/Filters/FilterSettings';
import { FilterTypeSelect } from 'features/controlLayers/components/Filters/FilterTypeSelect';
import { $canvasManager } from 'features/controlLayers/konva/CanvasManager';
import { $filterConfig, $filteringEntity, $isProcessingFilter } from 'features/controlLayers/store/canvasV2Slice';
import { type FilterConfig, IMAGE_FILTERS } from 'features/controlLayers/store/types';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCheckBold, PiShootingStarBold, PiXBold } from 'react-icons/pi';

export const Filter = memo(() => {
  const { t } = useTranslation();
  const filteringEntity = useStore($filteringEntity);
  const filterConfig = useStore($filterConfig);
  const isProcessingFilter = useStore($isProcessingFilter);

  const previewFilter = useCallback(() => {
    if (!filteringEntity) {
      return;
    }
    const canvasManager = $canvasManager.get();
    if (!canvasManager) {
      return;
    }
    const entity = canvasManager.stateApi.getEntity(filteringEntity);
    if (!entity || (entity.type !== 'raster_layer' && entity.type !== 'control_layer')) {
      return;
    }
    entity.adapter.filter.previewFilter();
  }, [filteringEntity]);

  const applyFilter = useCallback(() => {
    if (!filteringEntity) {
      return;
    }
    const canvasManager = $canvasManager.get();
    if (!canvasManager) {
      return;
    }
    const entity = canvasManager.stateApi.getEntity(filteringEntity);
    if (!entity || (entity.type !== 'raster_layer' && entity.type !== 'control_layer')) {
      return;
    }
    entity.adapter.filter.applyFilter();
  }, [filteringEntity]);

  const cancelFilter = useCallback(() => {
    if (!filteringEntity) {
      return;
    }
    const canvasManager = $canvasManager.get();
    if (!canvasManager) {
      return;
    }
    const entity = canvasManager.stateApi.getEntity(filteringEntity);
    if (!entity || (entity.type !== 'raster_layer' && entity.type !== 'control_layer')) {
      return;
    }
    entity.adapter.filter.cancelFilter();
  }, [filteringEntity]);

  const onChangeFilterConfig = useCallback((filterConfig: FilterConfig) => {
    $filterConfig.set(filterConfig);
  }, []);

  const onChangeFilterType = useCallback((filterType: FilterConfig['type']) => {
    $filterConfig.set(IMAGE_FILTERS[filterType].buildDefaults());
  }, []);

  if (!filteringEntity || !filterConfig) {
    return null;
  }

  return (
    <Flex
      bg="base.800"
      borderRadius="base"
      p={4}
      flexDir="column"
      gap={4}
      w={420}
      h="auto"
      shadow="dark-lg"
      transitionProperty="height"
      transitionDuration="normal"
    >
      <Heading size="md" color="base.300" userSelect="none">
        {t('controlLayers.filter.filter')}
      </Heading>
      <FilterTypeSelect filterType={filterConfig.type} onChange={onChangeFilterType} />
      <FilterSettings filterConfig={filterConfig} onChange={onChangeFilterConfig} />
      <ButtonGroup isAttached={false} size="sm" alignSelf="self-end">
        <Button
          leftIcon={<PiShootingStarBold />}
          onClick={previewFilter}
          isLoading={isProcessingFilter}
          loadingText={t('controlLayers.filter.preview')}
        >
          {t('controlLayers.filter.preview')}
        </Button>
        <Button
          leftIcon={<PiCheckBold />}
          onClick={applyFilter}
          isLoading={isProcessingFilter}
          loadingText={t('controlLayers.filter.apply')}
        >
          {t('controlLayers.filter.apply')}
        </Button>
        <Button
          leftIcon={<PiXBold />}
          onClick={cancelFilter}
          isLoading={isProcessingFilter}
          loadingText={t('controlLayers.filter.cancel')}
        >
          {t('controlLayers.filter.cancel')}
        </Button>
      </ButtonGroup>
    </Flex>
  );
});

Filter.displayName = 'Filter';
