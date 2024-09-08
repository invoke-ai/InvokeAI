import { Flex, Text } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { IconSwitch } from 'common/components/IconSwitch';
import {
  selectCanvasSettingsSlice,
  settingsSendToCanvasChanged,
} from 'features/controlLayers/store/canvasSettingsSlice';
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

const selectSendToCanvas = createSelector(selectCanvasSettingsSlice, (canvasSettings) => canvasSettings.sendToCanvas);

export const CanvasSendToToggle = memo(() => {
  const dispatch = useAppDispatch();
  const sendToCanvas = useAppSelector(selectSendToCanvas);

  const onChange = useCallback(
    (isChecked: boolean) => {
      dispatch(settingsSendToCanvasChanged(isChecked));
    },
    [dispatch]
  );

  return (
    <IconSwitch
      isChecked={sendToCanvas}
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
