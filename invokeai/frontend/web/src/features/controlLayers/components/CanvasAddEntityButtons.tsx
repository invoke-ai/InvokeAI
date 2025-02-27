import { Button, Flex, Heading } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { InformationalPopover } from 'common/components/InformationalPopover/InformationalPopover';
import {
  useAddControlLayer,
  useAddGlobalReferenceImage,
  useAddInpaintMask,
  useAddRasterLayer,
  useAddRegionalGuidance,
  useAddRegionalReferenceImage,
} from 'features/controlLayers/hooks/addLayerHooks';
import { selectIsFLUX, selectIsSD3 } from 'features/controlLayers/store/paramsSlice';
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
  const isSD3 = useAppSelector(selectIsSD3);

  return (
    <Flex w="full" h="full" justifyContent="center" gap={4}>
      <Flex position="relative" flexDir="column" gap={4} top="20%">
        <Flex flexDir="column" justifyContent="flex-start" gap={2}>
          <Heading size="xs">{t('controlLayers.global')}</Heading>
          <InformationalPopover feature="globalReferenceImage">
            <Button
              size="sm"
              variant="ghost"
              justifyContent="flex-start"
              leftIcon={<PiPlusBold />}
              onClick={addGlobalReferenceImage}
              isDisabled={isSD3}
            >
              {t('controlLayers.globalReferenceImage')}
            </Button>
          </InformationalPopover>
        </Flex>
        <Flex flexDir="column" gap={2}>
          <Heading size="xs">{t('controlLayers.regional')}</Heading>
          <InformationalPopover feature="inpainting">
            <Button
              size="sm"
              variant="ghost"
              justifyContent="flex-start"
              leftIcon={<PiPlusBold />}
              onClick={addInpaintMask}
            >
              {t('controlLayers.inpaintMask')}
            </Button>
          </InformationalPopover>
          <InformationalPopover feature="regionalGuidance">
            <Button
              size="sm"
              variant="ghost"
              justifyContent="flex-start"
              leftIcon={<PiPlusBold />}
              onClick={addRegionalGuidance}
              isDisabled={isSD3}
            >
              {t('controlLayers.regionalGuidance')}
            </Button>
          </InformationalPopover>
          <InformationalPopover feature="regionalReferenceImage">
            <Button
              size="sm"
              variant="ghost"
              justifyContent="flex-start"
              leftIcon={<PiPlusBold />}
              onClick={addRegionalReferenceImage}
              isDisabled={isFLUX || isSD3}
            >
              {t('controlLayers.regionalReferenceImage')}
            </Button>
          </InformationalPopover>
        </Flex>
        <Flex flexDir="column" justifyContent="flex-start" gap={2}>
          <Heading size="xs">{t('controlLayers.layer_other')}</Heading>
          <InformationalPopover feature="controlNet">
            <Button
              size="sm"
              variant="ghost"
              justifyContent="flex-start"
              leftIcon={<PiPlusBold />}
              onClick={addControlLayer}
              isDisabled={isSD3}
            >
              {t('controlLayers.controlLayer')}
            </Button>
          </InformationalPopover>
          <InformationalPopover feature="rasterLayer">
            <Button
              size="sm"
              variant="ghost"
              justifyContent="flex-start"
              leftIcon={<PiPlusBold />}
              onClick={addRasterLayer}
            >
              {t('controlLayers.rasterLayer')}
            </Button>
          </InformationalPopover>
        </Flex>
      </Flex>
    </Flex>
  );
});

CanvasAddEntityButtons.displayName = 'CanvasAddEntityButtons';
