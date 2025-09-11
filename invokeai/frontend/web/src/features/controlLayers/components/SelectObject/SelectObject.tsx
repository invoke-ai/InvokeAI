import { Flex, Heading, Spacer } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useFocusRegion, useIsRegionFocused } from 'common/hooks/focus';
import { CanvasAutoProcessSwitch } from 'features/controlLayers/components/CanvasAutoProcessSwitch';
import { CanvasOperationIsolatedLayerPreviewSwitch } from 'features/controlLayers/components/CanvasOperationIsolatedLayerPreviewSwitch';
import { SelectObjectActionButtons } from 'features/controlLayers/components/SelectObject/SelectObjectActionButtons';
import { SelectObjectInfoTooltip } from 'features/controlLayers/components/SelectObject/SelectObjectInfoTooltip';
import { SelectObjectInputTypeButtons } from 'features/controlLayers/components/SelectObject/SelectObjectInputTypeButtons';
import { SelectObjectInvert } from 'features/controlLayers/components/SelectObject/SelectObjectInvert';
import { SelectObjectPointType } from 'features/controlLayers/components/SelectObject/SelectObjectPointType';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import type { CanvasEntityAdapterControlLayer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterControlLayer';
import type { CanvasEntityAdapterRasterLayer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterRasterLayer';
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import { memo, useRef } from 'react';
import { useTranslation } from 'react-i18next';

import { SelectObjectModel } from './SelectObjectModel';
import { SelectObjectPrompt } from './SelectObjectPrompt';

const SelectObjectContent = memo(
  ({ adapter }: { adapter: CanvasEntityAdapterRasterLayer | CanvasEntityAdapterControlLayer }) => {
    const { t } = useTranslation();
    const ref = useRef<HTMLDivElement>(null);
    useFocusRegion('canvas', ref, { focusOnMount: true });
    const isCanvasFocused = useIsRegionFocused('canvas');
    const isProcessing = useStore(adapter.segmentAnything.$isProcessing);
    const inputType = useStore(adapter.segmentAnything.$inputType);

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
        <Flex w="full" gap={4} alignItems="center">
          <Flex gap={2}>
            <Heading size="md" color="base.300" userSelect="none">
              {t('controlLayers.selectObject.selectObject')}
            </Heading>
            <SelectObjectInfoTooltip />
          </Flex>
          <Spacer />
          <CanvasAutoProcessSwitch />
          <CanvasOperationIsolatedLayerPreviewSwitch />
        </Flex>

        <Flex w="full" justifyContent="space-between" py={2}>
          <SelectObjectInputTypeButtons adapter={adapter} />
          <SelectObjectInvert adapter={adapter} />
        </Flex>

        {inputType === 'visual' && <SelectObjectPointType adapter={adapter} />}
        {inputType === 'prompt' && <SelectObjectPrompt adapter={adapter} />}
        <SelectObjectModel adapter={adapter} />
        <SelectObjectActionButtons adapter={adapter} />
      </Flex>
    );
  }
);

SelectObjectContent.displayName = 'SegmentAnythingContent';

export const SelectObject = memo(() => {
  const canvasManager = useCanvasManager();
  const adapter = useStore(canvasManager.stateApi.$segmentingAdapter);

  if (!adapter) {
    return null;
  }

  return <SelectObjectContent adapter={adapter} />;
});

SelectObject.displayName = 'SelectObject';
