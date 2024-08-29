import { Flex, Text } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { IconSwitch } from 'common/components/IconSwitch';
import { selectIsComposing, sessionSendToCanvasChanged } from 'features/controlLayers/store/canvasSessionSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiImageBold, PiPaintBrushBold } from 'react-icons/pi';

const TooltipSendToGallery = memo(() => {
  const { t } = useTranslation();

  return (
    <Flex flexDir="column">
      <Text fontWeight="semibold">{t('controlLayers.sendToGallery')}</Text>
      <Text fontWeight="normal">{t('controlLayers.sendToGalleryDesc')}</Text>
    </Flex>
  );
});

TooltipSendToGallery.displayName = 'TooltipSendToGallery';

const TooltipSendToCanvas = memo(() => {
  const { t } = useTranslation();

  return (
    <Flex flexDir="column">
      <Text fontWeight="semibold">{t('controlLayers.sendToCanvas')}</Text>
      <Text fontWeight="normal">{t('controlLayers.sendToCanvasDesc')}</Text>
    </Flex>
  );
});

TooltipSendToCanvas.displayName = 'TooltipSendToCanvas';

export const CanvasSendToToggle = memo(() => {
  const dispatch = useAppDispatch();
  const isComposing = useAppSelector(selectIsComposing);

  const onChange = useCallback(
    (isChecked: boolean) => {
      dispatch(sessionSendToCanvasChanged(isChecked));
    },
    [dispatch]
  );

  return (
    <IconSwitch
      isChecked={isComposing}
      onChange={onChange}
      iconUnchecked={<PiImageBold />}
      tooltipUnchecked={<TooltipSendToGallery />}
      iconChecked={<PiPaintBrushBold />}
      tooltipChecked={<TooltipSendToCanvas />}
      ariaLabel="Toggle canvas mode"
    />
  );
});

CanvasSendToToggle.displayName = 'CanvasSendToToggle';
