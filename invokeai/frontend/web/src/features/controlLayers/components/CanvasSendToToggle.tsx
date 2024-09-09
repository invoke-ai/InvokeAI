import {
  Button,
  Flex,
  Icon,
  Popover,
  PopoverArrow,
  PopoverBody,
  PopoverContent,
  PopoverTrigger,
  Text,
} from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { selectSendToCanvas, settingsSendToCanvasChanged } from 'features/controlLayers/store/canvasSettingsSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCaretDownBold, PiCheckBold } from 'react-icons/pi';

export const CanvasSendToToggle = memo(() => {
  const { t } = useTranslation();
  const sendToCanvas = useAppSelector(selectSendToCanvas);
  const dispatch = useAppDispatch();

  const enableSendToCanvas = useCallback(() => {
    dispatch(settingsSendToCanvasChanged(true));
  }, [dispatch]);

  const disableSendToCanvas = useCallback(() => {
    dispatch(settingsSendToCanvasChanged(false));
  }, [dispatch]);

  return (
    <Popover isLazy>
      <PopoverTrigger>
        <Button
          size="sm"
          variant="link"
          data-testid="toggle-viewer-menu-button"
          pointerEvents="auto"
          rightIcon={<PiCaretDownBold />}
        >
          {sendToCanvas ? t('controlLayers.sendingToCanvas') : t('controlLayers.sendingToGallery')}
        </Button>
      </PopoverTrigger>
      <PopoverContent p={2} pointerEvents="auto">
        <PopoverArrow />
        <PopoverBody>
          <Flex flexDir="column">
            <Button onClick={disableSendToCanvas} variant="ghost" h="auto" w="auto" p={2}>
              <Flex gap={2} w="full">
                <Icon as={PiCheckBold} visibility={!sendToCanvas ? 'visible' : 'hidden'} />
                <Flex flexDir="column" gap={2} alignItems="flex-start">
                  <Text fontWeight="semibold">{t('controlLayers.sendToGallery')}</Text>
                  <Text fontWeight="normal" variant="subtext">
                    {t('controlLayers.sendToGalleryDesc')}
                  </Text>
                </Flex>
              </Flex>
            </Button>
            <Button onClick={enableSendToCanvas} variant="ghost" h="auto" w="auto" p={2}>
              <Flex gap={2} w="full">
                <Icon as={PiCheckBold} visibility={sendToCanvas ? 'visible' : 'hidden'} />
                <Flex flexDir="column" gap={2} alignItems="flex-start">
                  <Text fontWeight="semibold">{t('controlLayers.sendToCanvas')}</Text>
                  <Text fontWeight="normal" variant="subtext">
                    {t('controlLayers.sendToCanvasDesc')}
                  </Text>
                </Flex>
              </Flex>
            </Button>
          </Flex>
        </PopoverBody>
      </PopoverContent>
    </Popover>
  );
});

CanvasSendToToggle.displayName = 'CanvasSendToToggle';
