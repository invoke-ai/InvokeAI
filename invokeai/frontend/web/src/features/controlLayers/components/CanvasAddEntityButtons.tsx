import { Button, Flex, Heading } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import {
  useAddControlLayer,
  useAddGlobalReferenceImage,
  useAddInpaintMask,
  useAddRasterLayer,
  useAddRegionalGuidance,
  useAddRegionalReferenceImage,
} from 'features/controlLayers/hooks/addLayerHooks';
import { selectIsFLUX } from 'features/controlLayers/store/paramsSlice';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiPlusBold } from 'react-icons/pi';

export const CanvasAddEntityButtons = memo(() => {
  const { t } = useTranslation();
  const addInpaintMask = useAddInpaintMask();
  const addRegionalGuidance = useAddRegionalGuidance();
  const addRasterLayer = useAddRasterLayer();
  const addControlLayer = useAddControlLayer();
  const addGlobalReferenceImage = useAddGlobalReferenceImage();
  const addRegionalReferenceImage = useAddRegionalReferenceImage();
  const isFLUX = useAppSelector(selectIsFLUX);

  return (
    <Flex w="full" h="full" justifyContent="center" gap={4}>
      <Flex position="relative" flexDir="column" gap={4} top="20%">
        <Flex flexDir="column" justifyContent="flex-start" gap={2}>
          <Heading size="xs">{t('controlLayers.global')}</Heading>
          <Button
            size="sm"
            variant="ghost"
            justifyContent="flex-start"
            leftIcon={<PiPlusBold />}
            onClick={addGlobalReferenceImage}
            isDisabled={isFLUX}
          >
            {t('controlLayers.globalReferenceImage')}
          </Button>
        </Flex>
        <Flex flexDir="column" gap={2}>
          <Heading size="xs">{t('controlLayers.regional')}</Heading>
          <Button
            size="sm"
            variant="ghost"
            justifyContent="flex-start"
            leftIcon={<PiPlusBold />}
            onClick={addInpaintMask}
          >
            {t('controlLayers.inpaintMask')}
          </Button>
          <Button
            size="sm"
            variant="ghost"
            justifyContent="flex-start"
            leftIcon={<PiPlusBold />}
            onClick={addRegionalGuidance}
            isDisabled={isFLUX}
          >
            {t('controlLayers.regionalGuidance')}
          </Button>
          <Button
            size="sm"
            variant="ghost"
            justifyContent="flex-start"
            leftIcon={<PiPlusBold />}
            onClick={addRegionalReferenceImage}
            isDisabled={isFLUX}
          >
            {t('controlLayers.regionalReferenceImage')}
          </Button>
        </Flex>
        <Flex flexDir="column" justifyContent="flex-start" gap={2}>
          <Heading size="xs">{t('controlLayers.layer_other')}</Heading>

          <Button
            size="sm"
            variant="ghost"
            justifyContent="flex-start"
            leftIcon={<PiPlusBold />}
            onClick={addControlLayer}
          >
            {t('controlLayers.controlLayer')}
          </Button>
          <Button
            size="sm"
            variant="ghost"
            justifyContent="flex-start"
            leftIcon={<PiPlusBold />}
            onClick={addRasterLayer}
          >
            {t('controlLayers.rasterLayer')}
          </Button>
        </Flex>
      </Flex>
    </Flex>
  );
});

CanvasAddEntityButtons.displayName = 'CanvasAddEntityButtons';
