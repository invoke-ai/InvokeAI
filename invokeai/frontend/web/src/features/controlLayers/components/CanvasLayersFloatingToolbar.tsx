import { ButtonGroup, Flex, IconButton, Tooltip } from '@invoke-ai/ui-library';
import {
  useAddControlLayer,
  useAddInpaintMask,
  useAddNewRegionalGuidanceWithARefImage,
  useAddRasterLayer,
  useAddRegionalGuidance,
} from 'features/controlLayers/hooks/addLayerHooks';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { useIsEntityTypeEnabled } from 'features/controlLayers/hooks/useIsEntityTypeEnabled';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import {
  PiEraserBold,
  PiImageBold,
  PiPaintBrushBold,
  PiRectangleBold,
  PiShapesFill,
} from 'react-icons/pi';

export const CanvasLayersFloatingToolbar = memo(() => {
  const { t } = useTranslation();
  const isBusy = useCanvasIsBusy();
  const addInpaintMask = useAddInpaintMask();
  const addRegionalGuidance = useAddRegionalGuidance();
  const addRegionalReferenceImage = useAddNewRegionalGuidanceWithARefImage();
  const addRasterLayer = useAddRasterLayer();
  const addControlLayer = useAddControlLayer();
  const isRegionalGuidanceEnabled = useIsEntityTypeEnabled('regional_guidance');
  const isControlLayerEnabled = useIsEntityTypeEnabled('control_layer');
  const isInpaintLayerEnabled = useIsEntityTypeEnabled('inpaint_mask');

  return (
    <Flex
      position="absolute"
      bottom={2}
      left="50%"
      transform="translateX(-50%)"
      bg="base.800"
      borderRadius="md"
      p={1}
      boxShadow="dark-lg"
      border="1px solid"
      borderColor="base.700"
    >
      <ButtonGroup variant="ghost" size="sm" spacing={1}>
        <Tooltip label={t('controlLayers.inpaintMask')} placement="top">
          <IconButton
            aria-label={t('controlLayers.inpaintMask')}
            icon={<PiEraserBold />}
            onClick={addInpaintMask}
            isDisabled={!isInpaintLayerEnabled || isBusy}
          />
        </Tooltip>
        <Tooltip label={t('controlLayers.regionalGuidance')} placement="top">
          <IconButton
            aria-label={t('controlLayers.regionalGuidance')}
            icon={<PiShapesFill />}
            onClick={addRegionalGuidance}
            isDisabled={!isRegionalGuidanceEnabled || isBusy}
          />
        </Tooltip>
        <Tooltip label={t('controlLayers.controlLayer')} placement="top">
          <IconButton
            aria-label={t('controlLayers.controlLayer')}
            icon={<PiRectangleBold />}
            onClick={addControlLayer}
            isDisabled={!isControlLayerEnabled || isBusy}
          />
        </Tooltip>
        <Tooltip label={t('controlLayers.rasterLayer')} placement="top">
          <IconButton
            aria-label={t('controlLayers.rasterLayer')}
            icon={<PiPaintBrushBold />}
            onClick={addRasterLayer}
            isDisabled={isBusy}
          />
        </Tooltip>
      </ButtonGroup>
    </Flex>
  );
});

CanvasLayersFloatingToolbar.displayName = 'CanvasLayersFloatingToolbar';