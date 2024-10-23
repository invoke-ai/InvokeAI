import { Button, ButtonGroup, Flex, Heading, Spacer } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppSelector } from 'app/store/storeHooks';
import { useFocusRegion, useIsRegionFocused } from 'common/hooks/focus';
import { CanvasAutoProcessSwitch } from 'features/controlLayers/components/CanvasAutoProcessSwitch';
import { CanvasOperationIsolatedLayerPreviewSwitch } from 'features/controlLayers/components/CanvasOperationIsolatedLayerPreviewSwitch';
import { FilterSettings } from 'features/controlLayers/components/Filters/FilterSettings';
import { FilterTypeSelect } from 'features/controlLayers/components/Filters/FilterTypeSelect';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import type { CanvasEntityAdapterControlLayer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterControlLayer';
import type { CanvasEntityAdapterRasterLayer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterRasterLayer';
import { selectAutoProcess } from 'features/controlLayers/store/canvasSettingsSlice';
import type { FilterConfig } from 'features/controlLayers/store/filters';
import { IMAGE_FILTERS } from 'features/controlLayers/store/filters';
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import { memo, useCallback, useMemo, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowsCounterClockwiseBold, PiCheckBold, PiShootingStarBold, PiXBold } from 'react-icons/pi';

const FilterContent = memo(
  ({ adapter }: { adapter: CanvasEntityAdapterRasterLayer | CanvasEntityAdapterControlLayer }) => {
    const { t } = useTranslation();
    const ref = useRef<HTMLDivElement>(null);
    useFocusRegion('canvas', ref, { focusOnMount: true });
    const config = useStore(adapter.filterer.$filterConfig);
    const isCanvasFocused = useIsRegionFocused('canvas');
    const isProcessing = useStore(adapter.filterer.$isProcessing);
    const hasProcessed = useStore(adapter.filterer.$hasProcessed);
    const autoProcess = useAppSelector(selectAutoProcess);

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

    const isValid = useMemo(() => {
      return IMAGE_FILTERS[config.type].validateConfig?.(config as never) ?? true;
    }, [config]);

    useRegisteredHotkeys({
      id: 'applyFilter',
      category: 'canvas',
      callback: adapter.filterer.apply,
      options: { enabled: !isProcessing && isCanvasFocused },
      dependencies: [adapter.filterer, isProcessing, isCanvasFocused],
    });

    useRegisteredHotkeys({
      id: 'cancelFilter',
      category: 'canvas',
      callback: adapter.filterer.cancel,
      options: { enabled: !isProcessing && isCanvasFocused },
      dependencies: [adapter.filterer, isProcessing, isCanvasFocused],
    });

    return (
      <Flex
        ref={ref}
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
        <Flex w="full" gap={4}>
          <Heading size="md" color="base.300" userSelect="none">
            {t('controlLayers.filter.filter')}
          </Heading>
          <Spacer />
          <CanvasAutoProcessSwitch />
          <CanvasOperationIsolatedLayerPreviewSwitch />
        </Flex>
        <FilterTypeSelect filterType={config.type} onChange={onChangeFilterType} />
        <FilterSettings filterConfig={config} onChange={onChangeFilterConfig} />
        <ButtonGroup isAttached={false} size="sm" w="full">
          <Button
            variant="ghost"
            leftIcon={<PiShootingStarBold />}
            onClick={adapter.filterer.processImmediate}
            isLoading={isProcessing}
            loadingText={t('controlLayers.filter.process')}
            isDisabled={!isValid || autoProcess}
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
            loadingText={t('controlLayers.filter.cancel')}
          >
            {t('controlLayers.filter.cancel')}
          </Button>
        </ButtonGroup>
      </Flex>
    );
  }
);

FilterContent.displayName = 'FilterContent';

export const Filter = () => {
  const canvasManager = useCanvasManager();
  const adapter = useStore(canvasManager.stateApi.$filteringAdapter);
  if (!adapter) {
    return null;
  }

  return <FilterContent adapter={adapter} />;
};

Filter.displayName = 'Filter';
