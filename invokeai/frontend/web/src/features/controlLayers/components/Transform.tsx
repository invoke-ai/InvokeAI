import { Button, ButtonGroup, Flex, Heading } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import {
  EntityIdentifierContext,
  useEntityIdentifierContext,
} from 'features/controlLayers/contexts/EntityIdentifierContext';
import { useEntityAdapter } from 'features/controlLayers/hooks/useEntityAdapter';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import {
  PiArrowsCounterClockwiseBold,
  PiCheckBold,
  PiFlipHorizontalFill,
  PiFlipVerticalFill,
  PiXBold,
} from 'react-icons/pi';

const TransformBox = memo(() => {
  const { t } = useTranslation();
  const entityIdentifier = useEntityIdentifierContext();
  const adapter = useEntityAdapter(entityIdentifier);
  const isProcessing = useStore(adapter.transformer.$isProcessing);

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
        {t('controlLayers.tool.transform')}
      </Heading>
      <ButtonGroup isAttached={false} size="sm" justifyContent="center">
        <Button
          leftIcon={<PiFlipHorizontalFill />}
          onClick={adapter.transformer.flipHorizontal}
          isLoading={isProcessing}
          loadingText={t('controlLayers.flipHorizontal')}
        >
          {t('controlLayers.flipHorizontal')}
        </Button>
        <Button
          leftIcon={<PiFlipVerticalFill />}
          onClick={adapter.transformer.flipVertical}
          isLoading={isProcessing}
          loadingText={t('controlLayers.flipVertical')}
        >
          {t('controlLayers.flipVertical')}
        </Button>
        <Button
          leftIcon={<PiArrowsCounterClockwiseBold />}
          onClick={adapter.transformer.resetTransform}
          isLoading={isProcessing}
          loadingText={t('controlLayers.reset')}
        >
          {t('accessibility.reset')}
        </Button>
      </ButtonGroup>
      <ButtonGroup isAttached={false} size="sm" alignSelf="self-end">
        <Button
          leftIcon={<PiCheckBold />}
          onClick={adapter.transformer.applyTransform}
          isLoading={isProcessing}
          loadingText={t('common.apply')}
          variant="ghost"
        >
          {t('common.apply')}
        </Button>
        <Button
          leftIcon={<PiXBold />}
          onClick={adapter.transformer.stopTransform}
          isLoading={isProcessing}
          loadingText={t('common.cancel')}
          variant="ghost"
        >
          {t('common.cancel')}
        </Button>
      </ButtonGroup>
    </Flex>
  );
});

TransformBox.displayName = 'Transform';

export const Transform = () => {
  const canvasManager = useCanvasManager();
  const transformingEntity = useStore(canvasManager.stateApi.$transformingEntity);

  if (!transformingEntity) {
    return null;
  }

  return (
    <EntityIdentifierContext.Provider value={transformingEntity}>
      <TransformBox />
    </EntityIdentifierContext.Provider>
  );
};
