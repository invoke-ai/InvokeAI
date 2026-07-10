import type { CanvasEngine } from '@workbench/canvas-engine/engine';
import type { CanvasRasterLayerContractV2 } from '@workbench/types';

import { HStack, Stack, Text } from '@chakra-ui/react';
import { socketHub } from '@workbench/backend/socketHub';
import { runUtilityGraph } from '@workbench/canvas-engine/backend/utilityQueue';
import { Button } from '@workbench/components/ui';
import { makeImageDurable } from '@workbench/gallery/api';
import {
  buildFilterDefaults,
  DEFAULT_CONTROL_FILTER_TYPE,
  getFilterDefinition,
} from '@workbench/generation/canvas/filterGraphs';
import { useCallback, useState, useSyncExternalStore } from 'react';
import { useTranslation } from 'react-i18next';

import { createLayerFilterController } from './layerFilterController';
import { LayerFilterControls } from './LayerFilterControls';
import { runLayerFilter } from './layerFilterRunner';

interface RasterLayerFilterSectionProps {
  engine: CanvasEngine | null;
  focusFilter: boolean;
  layer: CanvasRasterLayerContractV2;
}

const defaultSettings = (): Record<string, unknown> => {
  const definition = getFilterDefinition(DEFAULT_CONTROL_FILTER_TYPE);
  return definition ? buildFilterDefaults(definition) : {};
};

export const RasterLayerFilterSection = ({ engine, focusFilter, layer }: RasterLayerFilterSectionProps) => {
  const { t } = useTranslation();
  const [filterType, setFilterType] = useState(DEFAULT_CONTROL_FILTER_TYPE);
  const [settings, setSettings] = useState<Record<string, unknown>>(defaultSettings);
  const [controller] = useState(() =>
    createLayerFilterController({
      clearPreview: () => engine?.setFilterPreview(layer.id, null),
      commit: (options) => engine?.commitRasterFilterResult(options) ?? Promise.resolve({ status: 'missing' as const }),
      exportPixels: () =>
        engine?.exportLayerPixels(layer.id, { applyAdjustments: true, includeDisabled: true }) ??
        Promise.resolve({ status: 'missing' as const }),
      makeDurable: makeImageDurable,
      runFilter: (options) => {
        if (!engine) {
          return Promise.reject(new Error('The canvas engine is unavailable.'));
        }
        const executorDeps = engine.getCompositeExecutorDeps();
        return runLayerFilter({
          ...options,
          deps: {
            encodeSurface: (surface) => executorDeps.backend.encodeSurface(surface, 'image/png'),
            runFilterGraph: async ({ graph, outputNodeId, signal }) => {
              const output = await runUtilityGraph({ graph, hub: socketHub, outputNodeId, signal });
              return { imageName: output.imageName };
            },
            uploadIntermediate: async (blob) => {
              const uploaded = await executorDeps.uploadImage(blob);
              return { imageName: uploaded.imageName };
            },
          },
        });
      },
      showPreview: (imageName, guard) =>
        engine?.setGuardedFilterPreview(layer.id, { imageName }, guard) ?? Promise.resolve('missing' as const),
    })
  );
  const { error, isRunning, preview } = useSyncExternalStore(
    controller.subscribe,
    controller.getSnapshot,
    controller.getSnapshot
  );

  const clearTransientPreview = useCallback(() => {
    controller.cancel();
  }, [controller]);

  const sectionRef = useCallback(
    (node: HTMLDivElement | null) => {
      if (!node) {
        return;
      }
      return () => {
        controller.cancel();
      };
    },
    [controller]
  );

  const handleFilterTypeChange = useCallback(
    (nextType: string) => {
      if (nextType === filterType) {
        return;
      }
      clearTransientPreview();
      setFilterType(nextType);
      const definition = getFilterDefinition(nextType);
      setSettings(definition ? buildFilterDefaults(definition) : {});
    },
    [clearTransientPreview, filterType]
  );

  const handleSettingsChange = useCallback(
    (nextSettings: Record<string, unknown>) => {
      clearTransientPreview();
      setSettings(nextSettings);
    },
    [clearTransientPreview]
  );

  const handlePreview = useCallback(
    () => void controller.preview(filterType, settings),
    [controller, filterType, settings]
  );
  const handleReplace = useCallback(() => void controller.commit('replace'), [controller]);
  const handleCopy = useCallback(() => void controller.commit('copy'), [controller]);
  const controlsDisabled = !engine || layer.isLocked || isRunning;
  const canCancel = isRunning || preview !== null || error !== null;
  const errorText = error
    ? error.key === 'commitFailure' || error.key === 'durabilityFailure' || error.key === 'graphFailure'
      ? t(`widgets.layers.rasterFilter.${error.key}`, { message: error.message })
      : t(`widgets.layers.rasterFilter.${error.key}`)
    : null;

  return (
    <Stack borderColor="border.subtle" borderTopWidth="1px" gap="2" pt="2" ref={sectionRef}>
      <Text color="fg.subtle" fontSize="2xs" fontWeight="700" textTransform="uppercase">
        {t('widgets.layers.rasterFilter.title')}
      </Text>
      <LayerFilterControls
        disabled={controlsDisabled}
        filterType={filterType}
        focusFilter={focusFilter}
        settings={settings}
        onFilterTypeChange={handleFilterTypeChange}
        onSettingsChange={handleSettingsChange}
      />
      <HStack gap="2" wrap="wrap">
        <Button
          disabled={controlsDisabled}
          loading={isRunning && preview === null}
          size="xs"
          variant="outline"
          onClick={handlePreview}
        >
          {t('widgets.layers.rasterFilter.preview')}
        </Button>
        {preview ? (
          <>
            <Button colorPalette="accent" disabled={isRunning} size="xs" onClick={handleReplace}>
              {t('widgets.layers.rasterFilter.replace')}
            </Button>
            <Button disabled={isRunning} size="xs" variant="outline" onClick={handleCopy}>
              {t('widgets.layers.rasterFilter.copy')}
            </Button>
          </>
        ) : null}
        <Button disabled={!canCancel} size="xs" variant="ghost" onClick={clearTransientPreview}>
          {t('widgets.layers.rasterFilter.cancel')}
        </Button>
      </HStack>
      {isRunning ? (
        <Text color="fg.subtle" fontSize="2xs">
          {t('widgets.layers.rasterFilter.running')}
        </Text>
      ) : null}
      {errorText ? (
        <Text color="fg.error" fontSize="2xs">
          {errorText}
        </Text>
      ) : null}
    </Stack>
  );
};
