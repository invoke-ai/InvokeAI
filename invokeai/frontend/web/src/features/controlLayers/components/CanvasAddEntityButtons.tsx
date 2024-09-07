import { Button, ButtonGroup, Flex } from '@invoke-ai/ui-library';
import {
  useAddControlLayer,
  useAddInpaintMask,
  useAddIPAdapter,
  useAddRasterLayer,
  useAddRegionalGuidance,
} from 'features/controlLayers/hooks/addLayerHooks';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiPlusBold } from 'react-icons/pi';

export const CanvasAddEntityButtons = memo(() => {
  const { t } = useTranslation();
  const addInpaintMask = useAddInpaintMask();
  const addRegionalGuidance = useAddRegionalGuidance();
  const addRasterLayer = useAddRasterLayer();
  const addControlLayer = useAddControlLayer();
  const addIPAdapter = useAddIPAdapter();

  return (
    <Flex flexDir="column" w="full" h="full" alignItems="center">
      <ButtonGroup position="relative" orientation="vertical" isAttached={false} top="20%">
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
          {t('controlLayers.globalIPAdapter')}
        </Button>
      </ButtonGroup>
    </Flex>
  );
});

CanvasAddEntityButtons.displayName = 'CanvasAddEntityButtons';
