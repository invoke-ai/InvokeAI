import { Button, ButtonGroup, Flex, FormControl, FormLabel, Heading, Spacer, Switch } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import type { CanvasEntityAdapter } from 'features/controlLayers/konva/CanvasEntity/types';
import {
  selectIsolatedTransformingPreview,
  settingsIsolatedTransformingPreviewToggled,
} from 'features/controlLayers/store/canvasSettingsSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowsCounterClockwiseBold, PiArrowsOutBold, PiCheckBold, PiXBold } from 'react-icons/pi';

const TransformBox = memo(({ adapter }: { adapter: CanvasEntityAdapter }) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const isProcessing = useStore(adapter.transformer.$isProcessing);
  const isolatedTransformingPreview = useAppSelector(selectIsolatedTransformingPreview);
  const onChangeIsolatedPreview = useCallback(() => {
    dispatch(settingsIsolatedTransformingPreviewToggled());
  }, [dispatch]);

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
      <Flex w="full" gap={4}>
        <Heading size="md" color="base.300" userSelect="none">
          {t('controlLayers.transform.transform')}
        </Heading>
        <Spacer />
        <FormControl w="min-content">
          <FormLabel m={0}>{t('controlLayers.settings.isolatedPreview')}</FormLabel>
          <Switch size="sm" isChecked={isolatedTransformingPreview} onChange={onChangeIsolatedPreview} />
        </FormControl>
      </Flex>
      <ButtonGroup isAttached={false} size="sm" w="full">
        <Button
          leftIcon={<PiArrowsOutBold />}
          onClick={adapter.transformer.fitProxyRectToBbox}
          isLoading={isProcessing}
          loadingText={t('controlLayers.transform.fitToBbox')}
          variant="ghost"
        >
          {t('controlLayers.transform.fitToBbox')}
        </Button>
        <Spacer />
        <Button
          leftIcon={<PiArrowsCounterClockwiseBold />}
          onClick={adapter.transformer.resetTransform}
          isLoading={isProcessing}
          loadingText={t('controlLayers.transform.reset')}
          variant="ghost"
        >
          {t('controlLayers.transform.reset')}
        </Button>
        <Button
          leftIcon={<PiCheckBold />}
          onClick={adapter.transformer.applyTransform}
          isLoading={isProcessing}
          loadingText={t('controlLayers.transform.apply')}
          variant="ghost"
        >
          {t('controlLayers.transform.apply')}
        </Button>
        <Button
          leftIcon={<PiXBold />}
          onClick={adapter.transformer.stopTransform}
          isLoading={isProcessing}
          loadingText={t('common.cancel')}
          variant="ghost"
        >
          {t('controlLayers.transform.cancel')}
        </Button>
      </ButtonGroup>
    </Flex>
  );
});

TransformBox.displayName = 'Transform';

export const Transform = () => {
  const canvasManager = useCanvasManager();
  const adapter = useStore(canvasManager.stateApi.$transformingAdapter);

  if (!adapter) {
    return null;
  }

  return <TransformBox adapter={adapter} />;
};
