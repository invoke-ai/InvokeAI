import { Button, ButtonGroup, Flex, FormControl, FormLabel, Heading, Spacer, Switch } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { FilterSettings } from 'features/controlLayers/components/Filters/FilterSettings';
import { FilterTypeSelect } from 'features/controlLayers/components/Filters/FilterTypeSelect';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import type { CanvasEntityAdapterControlLayer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterControlLayer';
import type { CanvasEntityAdapterRasterLayer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterRasterLayer';
import {
  selectAutoProcessFilter,
  settingsAutoProcessFilterToggled,
} from 'features/controlLayers/store/canvasSettingsSlice';
import type { FilterConfig } from 'features/controlLayers/store/filters';
import { IMAGE_FILTERS } from 'features/controlLayers/store/filters';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowsCounterClockwiseBold, PiCheckBold, PiShootingStarBold, PiXBold } from 'react-icons/pi';

const FilterBox = memo(({ adapter }: { adapter: CanvasEntityAdapterRasterLayer | CanvasEntityAdapterControlLayer }) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const config = useStore(adapter.filterer.$filterConfig);
  const isProcessing = useStore(adapter.filterer.$isProcessing);
  const hasProcessed = useStore(adapter.filterer.$hasProcessed);
  const autoProcessFilter = useAppSelector(selectAutoProcessFilter);

  const onChangeFilterConfig = useCallback(
    (filterConfig: FilterConfig) => {
      adapter.filterer.$filterConfig.set(filterConfig);
    },
    [adapter.filterer.$filterConfig]
  );

  const onChangeFilterType = useCallback(
    (filterType: FilterConfig['type']) => {
      adapter.filterer.$filterConfig.set(IMAGE_FILTERS[filterType].buildDefaults());
    },
    [adapter.filterer.$filterConfig]
  );

  const onChangeAutoProcessFilter = useCallback(() => {
    dispatch(settingsAutoProcessFilterToggled());
  }, [dispatch]);

  const isValid = useMemo(() => {
    return IMAGE_FILTERS[config.type].validateConfig?.(config as never) ?? true;
  }, [config]);

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
      <Flex w="full">
        <Heading size="md" color="base.300" userSelect="none">
          {t('controlLayers.filter.filter')}
        </Heading>
        <Spacer />
        <FormControl w="min-content">
          <FormLabel m={0}>{t('controlLayers.filter.autoProcess')}</FormLabel>
          <Switch size="sm" isChecked={autoProcessFilter} onChange={onChangeAutoProcessFilter} />
        </FormControl>
      </Flex>
      <FilterTypeSelect filterType={config.type} onChange={onChangeFilterType} />
      <FilterSettings filterConfig={config} onChange={onChangeFilterConfig} />
      <ButtonGroup isAttached={false} size="sm" w="full">
        <Button
          variant="ghost"
          leftIcon={<PiShootingStarBold />}
          onClick={adapter.filterer.process}
          isLoading={isProcessing}
          loadingText={t('controlLayers.filter.process')}
          isDisabled={!isValid || autoProcessFilter}
        >
          {t('controlLayers.filter.process')}
        </Button>
        <Spacer />
        <Button
          leftIcon={<PiArrowsCounterClockwiseBold />}
          onClick={adapter.filterer.reset}
          isLoading={isProcessing}
          loadingText={t('controlLayers.filter.reset')}
          variant="ghost"
        >
          {t('controlLayers.filter.reset')}
        </Button>
        <Button
          variant="ghost"
          leftIcon={<PiCheckBold />}
          onClick={adapter.filterer.apply}
          isLoading={isProcessing}
          loadingText={t('controlLayers.filter.apply')}
          isDisabled={!isValid || !hasProcessed}
        >
          {t('controlLayers.filter.apply')}
        </Button>
        <Button
          variant="ghost"
          leftIcon={<PiXBold />}
          onClick={adapter.filterer.cancel}
          isLoading={isProcessing}
          loadingText={t('controlLayers.filter.cancel')}
        >
          {t('controlLayers.filter.cancel')}
        </Button>
      </ButtonGroup>
    </Flex>
  );
});

FilterBox.displayName = 'FilterBox';

export const Filter = () => {
  const canvasManager = useCanvasManager();
  const adapter = useStore(canvasManager.stateApi.$filteringAdapter);
  if (!adapter) {
    return null;
  }

  return <FilterBox adapter={adapter} />;
};

Filter.displayName = 'Filter';
