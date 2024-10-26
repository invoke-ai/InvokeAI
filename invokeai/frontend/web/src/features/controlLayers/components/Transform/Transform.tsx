import { Button, ButtonGroup, Flex, Heading, Spacer, Spinner } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useFocusRegion, useIsRegionFocused } from 'common/hooks/focus';
import { CanvasOperationIsolatedLayerPreviewSwitch } from 'features/controlLayers/components/CanvasOperationIsolatedLayerPreviewSwitch';
import { TransformFitToBboxButtons } from 'features/controlLayers/components/Transform/TransformFitToBboxButtons';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import type { CanvasEntityAdapter } from 'features/controlLayers/konva/CanvasEntity/types';
import { useRegisteredHotkeys } from 'features/system/components/HotkeysModal/useHotkeyData';
import { memo, useRef } from 'react';
import { useTranslation } from 'react-i18next';

const TransformContent = memo(({ adapter }: { adapter: CanvasEntityAdapter }) => {
  const { t } = useTranslation();
  const ref = useRef<HTMLDivElement>(null);
  useFocusRegion('canvas', ref, { focusOnMount: true });
  const isCanvasFocused = useIsRegionFocused('canvas');
  const isProcessing = useStore(adapter.transformer.$isProcessing);
  const silentTransform = useStore(adapter.transformer.$silentTransform);

  useRegisteredHotkeys({
    id: 'applyTransform',
    category: 'canvas',
    callback: adapter.transformer.applyTransform,
    options: { enabled: !isProcessing && isCanvasFocused },
    dependencies: [adapter.transformer, isProcessing, isCanvasFocused],
  });

  useRegisteredHotkeys({
    id: 'cancelTransform',
    category: 'canvas',
    callback: adapter.transformer.stopTransform,
    options: { enabled: !isProcessing && isCanvasFocused },
    dependencies: [adapter.transformer, isProcessing, isCanvasFocused],
  });

  if (silentTransform) {
    return null;
  }

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
          {t('controlLayers.transform.transform')}
        </Heading>
        <Spacer />
        <CanvasOperationIsolatedLayerPreviewSwitch />
      </Flex>

      <TransformFitToBboxButtons adapter={adapter} />

      <ButtonGroup isAttached={false} size="sm" w="full" alignItems="center">
        {isProcessing && <Spinner ms={3} boxSize={5} color="base.600" />}
        <Spacer />
        <Button
          onClick={adapter.transformer.resetTransform}
          isDisabled={isProcessing}
          loadingText={t('controlLayers.transform.reset')}
          variant="ghost"
        >
          {t('controlLayers.transform.reset')}
        </Button>
        <Button
          onClick={adapter.transformer.applyTransform}
          isDisabled={isProcessing}
          loadingText={t('controlLayers.transform.apply')}
          variant="ghost"
        >
          {t('controlLayers.transform.apply')}
        </Button>
        <Button
          onClick={adapter.transformer.stopTransform}
          isDisabled={isProcessing}
          loadingText={t('common.cancel')}
          variant="ghost"
        >
          {t('controlLayers.transform.cancel')}
        </Button>
      </ButtonGroup>
    </Flex>
  );
});

TransformContent.displayName = 'TransformContent';

export const Transform = () => {
  const canvasManager = useCanvasManager();
  const adapter = useStore(canvasManager.stateApi.$transformingAdapter);

  if (!adapter) {
    return null;
  }

  return <TransformContent adapter={adapter} />;
};
