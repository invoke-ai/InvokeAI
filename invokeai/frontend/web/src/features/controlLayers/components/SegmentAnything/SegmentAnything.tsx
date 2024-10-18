import { Button, ButtonGroup, Flex, Heading, Spacer } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useFocusRegion, useIsRegionFocused } from 'common/hooks/focus';
import { SegmentAnythingPointType } from 'features/controlLayers/components/SegmentAnything/SegmentAnythingPointType';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import type { CanvasEntityAdapterControlLayer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterControlLayer';
import type { CanvasEntityAdapterRasterLayer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterRasterLayer';
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import { memo, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowsCounterClockwiseBold, PiCheckBold, PiXBold } from 'react-icons/pi';

const SegmentAnythingContent = memo(
  ({ adapter }: { adapter: CanvasEntityAdapterRasterLayer | CanvasEntityAdapterControlLayer }) => {
    const { t } = useTranslation();
    const ref = useRef<HTMLDivElement>(null);
    useFocusRegion('canvas', ref, { focusOnMount: true });
    const isCanvasFocused = useIsRegionFocused('canvas');
    const isProcessing = useStore(adapter.segmentAnything.$isProcessing);

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
            {t('controlLayers.segment.segment')}
          </Heading>
        </Flex>

        <SegmentAnythingPointType adapter={adapter} />

        <ButtonGroup isAttached={false} size="sm" w="full">
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
          <Button
            leftIcon={<PiCheckBold />}
            onClick={adapter.segmentAnything.apply}
            isLoading={isProcessing}
            loadingText={t('controlLayers.segment.apply')}
            variant="ghost"
          >
            {t('controlLayers.segment.apply')}
          </Button>
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
