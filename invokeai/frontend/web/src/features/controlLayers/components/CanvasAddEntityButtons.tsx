import { Button, ButtonGroup, Flex } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import {
  controlLayerAdded,
  inpaintMaskAdded,
  ipaAdded,
  rasterLayerAdded,
  rgAdded,
} from 'features/controlLayers/store/canvasSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiPlusBold } from 'react-icons/pi';

export const CanvasAddEntityButtons = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const addInpaintMask = useCallback(() => {
    dispatch(inpaintMaskAdded({ isSelected: true }));
  }, [dispatch]);
  const addRegionalGuidance = useCallback(() => {
    dispatch(rgAdded({ isSelected: true }));
  }, [dispatch]);
  const addRasterLayer = useCallback(() => {
    dispatch(rasterLayerAdded({ isSelected: true }));
  }, [dispatch]);
  const addControlLayer = useCallback(() => {
    dispatch(controlLayerAdded({ isSelected: true }));
  }, [dispatch]);
  const addIPAdapter = useCallback(() => {
    dispatch(ipaAdded({ isSelected: true }));
  }, [dispatch]);

  return (
    <Flex flexDir="column" w="full" h="full" alignItems="center" justifyContent="center">
      <ButtonGroup orientation="vertical" isAttached={false}>
        <Button variant="ghost" justifyContent="flex-start" leftIcon={<PiPlusBold />} onClick={addInpaintMask}>
          {t('controlLayers.inpaintMask')}
        </Button>
        <Button variant="ghost" justifyContent="flex-start" leftIcon={<PiPlusBold />} onClick={addRegionalGuidance}>
          {t('controlLayers.regionalGuidance')}
        </Button>
        <Button variant="ghost" justifyContent="flex-start" leftIcon={<PiPlusBold />} onClick={addRasterLayer}>
          {t('controlLayers.rasterLayer')}
        </Button>
        <Button variant="ghost" justifyContent="flex-start" leftIcon={<PiPlusBold />} onClick={addControlLayer}>
          {t('controlLayers.controlLayer')}
        </Button>
        <Button variant="ghost" justifyContent="flex-start" leftIcon={<PiPlusBold />} onClick={addIPAdapter}>
          {t('controlLayers.ipAdapter')}
        </Button>
      </ButtonGroup>
    </Flex>
  );
});

CanvasAddEntityButtons.displayName = 'CanvasAddEntityButtons';
