import {
  Button,
  ButtonGroup,
  Flex,
  Heading,
  Menu,
  MenuButton,
  MenuItem,
  MenuList,
  Spacer,
} from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppSelector } from 'app/store/storeHooks';
import { useFocusRegion, useIsRegionFocused } from 'common/hooks/focus';
import { CanvasAutoProcessSwitch } from 'features/controlLayers/components/CanvasAutoProcessSwitch';
import { CanvasOperationIsolatedLayerPreviewSwitch } from 'features/controlLayers/components/CanvasOperationIsolatedLayerPreviewSwitch';
import { SegmentAnythingPointType } from 'features/controlLayers/components/SegmentAnything/SegmentAnythingPointType';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import type { CanvasEntityAdapterControlLayer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterControlLayer';
import type { CanvasEntityAdapterRasterLayer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterRasterLayer';
import { selectAutoProcess } from 'features/controlLayers/store/canvasSettingsSlice';
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import { memo, useCallback, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowsCounterClockwiseBold, PiFloppyDiskBold, PiStarBold, PiXBold } from 'react-icons/pi';

const SegmentAnythingContent = memo(
  ({ adapter }: { adapter: CanvasEntityAdapterRasterLayer | CanvasEntityAdapterControlLayer }) => {
    const { t } = useTranslation();
    const ref = useRef<HTMLDivElement>(null);
    useFocusRegion('canvas', ref, { focusOnMount: true });
    const isCanvasFocused = useIsRegionFocused('canvas');
    const isProcessing = useStore(adapter.segmentAnything.$isProcessing);
    const hasPoints = useStore(adapter.segmentAnything.$hasPoints);
    const hasImageState = useStore(adapter.segmentAnything.$hasImageState);
    const autoProcess = useAppSelector(selectAutoProcess);

    const saveAsInpaintMask = useCallback(() => {
      adapter.segmentAnything.saveAs('inpaint_mask');
    }, [adapter.segmentAnything]);

    const saveAsRegionalGuidance = useCallback(() => {
      adapter.segmentAnything.saveAs('regional_guidance');
    }, [adapter.segmentAnything]);

    const saveAsRasterLayer = useCallback(() => {
      adapter.segmentAnything.saveAs('raster_layer');
    }, [adapter.segmentAnything]);

    const saveAsControlLayer = useCallback(() => {
      adapter.segmentAnything.saveAs('control_layer');
    }, [adapter.segmentAnything]);

    useRegisteredHotkeys({
      id: 'applySegmentAnything',
      category: 'canvas',
      callback: adapter.segmentAnything.apply,
      options: { enabled: !isProcessing && isCanvasFocused },
      dependencies: [adapter.segmentAnything, isProcessing, isCanvasFocused],
    });

    useRegisteredHotkeys({
      id: 'cancelSegmentAnything',
      category: 'canvas',
      callback: adapter.segmentAnything.cancel,
      options: { enabled: !isProcessing && isCanvasFocused },
      dependencies: [adapter.segmentAnything, isProcessing, isCanvasFocused],
    });

    return (
      <Flex
        ref={ref}
        bg="base.800"
        borderRadius="base"
        p={4}
        flexDir="column"
        gap={4}
        minW={420}
        h="auto"
        shadow="dark-lg"
        transitionProperty="height"
        transitionDuration="normal"
      >
        <Flex w="full" gap={4}>
          <Heading size="md" color="base.300" userSelect="none">
            {t('controlLayers.segment.autoMask')}
          </Heading>
          <Spacer />
          <CanvasAutoProcessSwitch />
          <CanvasOperationIsolatedLayerPreviewSwitch />
        </Flex>

        <SegmentAnythingPointType adapter={adapter} />

        <ButtonGroup isAttached={false} size="sm" w="full">
          <Button
            leftIcon={<PiStarBold />}
            onClick={adapter.segmentAnything.processImmediate}
            isLoading={isProcessing}
            loadingText={t('controlLayers.segment.process')}
            variant="ghost"
            isDisabled={!hasPoints || autoProcess}
          >
            {t('controlLayers.segment.process')}
          </Button>
          <Spacer />
          <Button
            leftIcon={<PiArrowsCounterClockwiseBold />}
            onClick={adapter.segmentAnything.reset}
            isLoading={isProcessing}
            loadingText={t('controlLayers.segment.reset')}
            variant="ghost"
          >
            {t('controlLayers.segment.reset')}
          </Button>
          <Menu>
            <MenuButton
              as={Button}
              leftIcon={<PiFloppyDiskBold />}
              isLoading={isProcessing}
              loadingText={t('controlLayers.segment.saveAs')}
              variant="ghost"
              isDisabled={!hasImageState}
            >
              {t('controlLayers.segment.saveAs')}
            </MenuButton>
            <MenuList>
              <MenuItem isDisabled={!hasImageState} onClick={saveAsInpaintMask}>
                {t('controlLayers.inpaintMask')}
              </MenuItem>
              <MenuItem isDisabled={!hasImageState} onClick={saveAsRegionalGuidance}>
                {t('controlLayers.regionalGuidance')}
              </MenuItem>
              <MenuItem isDisabled={!hasImageState} onClick={saveAsControlLayer}>
                {t('controlLayers.controlLayer')}
              </MenuItem>
              <MenuItem isDisabled={!hasImageState} onClick={saveAsRasterLayer}>
                {t('controlLayers.rasterLayer')}
              </MenuItem>
            </MenuList>
          </Menu>
          <Button
            leftIcon={<PiXBold />}
            onClick={adapter.segmentAnything.cancel}
            isLoading={isProcessing}
            loadingText={t('common.cancel')}
            variant="ghost"
          >
            {t('controlLayers.segment.cancel')}
          </Button>
        </ButtonGroup>
      </Flex>
    );
  }
);

SegmentAnythingContent.displayName = 'SegmentAnythingContent';

export const SegmentAnything = () => {
  const canvasManager = useCanvasManager();
  const adapter = useStore(canvasManager.stateApi.$segmentingAdapter);

  if (!adapter) {
    return null;
  }

  return <SegmentAnythingContent adapter={adapter} />;
};
