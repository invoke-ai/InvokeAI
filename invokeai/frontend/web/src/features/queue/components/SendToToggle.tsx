import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Box, Flex, IconButton, Text, Tooltip, useToken } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  selectCanvasSettingsSlice,
  settingsSendToCanvasChanged,
} from 'features/controlLayers/store/canvasSettingsSlice';
import { memo, useCallback, useMemo } from 'react';
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

const getSx = (padding: string | number): SystemStyleObject => ({
  transition: 'left 0.1s ease-in-out, transform 0.1s ease-in-out',
  '&[data-checked="true"]': {
    left: `calc(100% - ${padding})`,
    transform: 'translateX(-100%)',
  },
  '&[data-checked="false"]': {
    left: padding,
    transform: 'translateX(0)',
  },
});

export const SendToToggle = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const sendToCanvas = useAppSelector(selectSendToCanvas);

  const gap = useToken('space', 1.5);
  const sx = useMemo(() => getSx(gap), [gap]);

  const onClickSendToGallery = useCallback(() => {
    if (!sendToCanvas) {
      return;
    }
    dispatch(settingsSendToCanvasChanged(false));
  }, [dispatch, sendToCanvas]);

  const onClickSendToCanvas = useCallback(() => {
    if (sendToCanvas) {
      return;
    }
    dispatch(settingsSendToCanvasChanged(true));
  }, [dispatch, sendToCanvas]);

  return (
    <Flex
      position="relative"
      bg="base.800"
      borderRadius="base"
      alignItems="center"
      justifyContent="center"
      h="full"
      p={gap}
      gap={gap}
    >
      <Box
        position="absolute"
        borderRadius="base"
        bg="invokeBlue.400"
        w={12}
        top={gap}
        bottom={gap}
        data-checked={sendToCanvas}
        sx={sx}
      />
      <Tooltip hasArrow label={<TooltipSendToGallery />}>
        <IconButton
          size="sm"
          fontSize={16}
          icon={<PiImageBold />}
          onClick={onClickSendToGallery}
          variant={!sendToCanvas ? 'solid' : 'ghost'}
          colorScheme={!sendToCanvas ? 'invokeBlue' : 'base'}
          aria-label={t('controlLayers.sendToGallery')}
          data-checked={!sendToCanvas}
          w={12}
          alignSelf="stretch"
          h="auto"
        />
      </Tooltip>
      <Tooltip hasArrow label={<TooltipSendToCanvas />}>
        <IconButton
          size="sm"
          fontSize={16}
          icon={<PiPaintBrushBold />}
          onClick={onClickSendToCanvas}
          variant={sendToCanvas ? 'solid' : 'ghost'}
          colorScheme={sendToCanvas ? 'invokeGreen' : 'base'}
          aria-label={t('controlLayers.sendToCanvas')}
          data-checked={sendToCanvas}
          w={12}
          alignSelf="stretch"
          h="auto"
        />
      </Tooltip>
    </Flex>
  );
});

SendToToggle.displayName = 'CanvasSendToToggle';
