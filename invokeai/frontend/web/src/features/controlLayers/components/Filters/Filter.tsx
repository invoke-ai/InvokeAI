import { Button, ButtonGroup, Flex, Heading } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { FilterSettings } from 'features/controlLayers/components/Filters/FilterSettings';
import { FilterTypeSelect } from 'features/controlLayers/components/Filters/FilterTypeSelect';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { type FilterConfig, IMAGE_FILTERS } from 'features/controlLayers/store/types';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCheckBold, PiShootingStarBold, PiXBold } from 'react-icons/pi';

export const Filter = memo(() => {
  const { t } = useTranslation();
  const canvasManager = useCanvasManager();
  const config = useStore(canvasManager.filter.$config);
  const isFiltering = useStore(canvasManager.filter.$isFiltering);
  const isProcessing = useStore(canvasManager.filter.$isProcessing);

  const onChangeFilterConfig = useCallback(
    (filterConfig: FilterConfig) => {
      canvasManager.filter.$config.set(filterConfig);
    },
    [canvasManager.filter.$config]
  );

  const onChangeFilterType = useCallback(
    (filterType: FilterConfig['type']) => {
      canvasManager.filter.$config.set(IMAGE_FILTERS[filterType].buildDefaults());
    },
    [canvasManager.filter.$config]
  );

  if (!isFiltering) {
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
      <FilterTypeSelect filterType={config.type} onChange={onChangeFilterType} />
      <FilterSettings filterConfig={config} onChange={onChangeFilterConfig} />
      <ButtonGroup isAttached={false} size="sm" alignSelf="self-end">
        <Button
          variant="ghost"
          leftIcon={<PiShootingStarBold />}
          onClick={canvasManager.filter.previewFilter}
          isLoading={isProcessing}
          loadingText={t('controlLayers.filter.preview')}
        >
          {t('controlLayers.filter.preview')}
        </Button>
        <Button
          variant="ghost"
          leftIcon={<PiCheckBold />}
          onClick={canvasManager.filter.applyFilter}
          isLoading={isProcessing}
          loadingText={t('controlLayers.filter.apply')}
        >
          {t('controlLayers.filter.apply')}
        </Button>
        <Button
          variant="ghost"
          leftIcon={<PiXBold />}
          onClick={canvasManager.filter.cancelFilter}
          isLoading={isProcessing}
          loadingText={t('controlLayers.filter.cancel')}
        >
          {t('controlLayers.filter.cancel')}
        </Button>
      </ButtonGroup>
    </Flex>
  );
});

Filter.displayName = 'Filter';
