import { IconButton, Menu, MenuButton, MenuGroup, MenuItem, MenuList } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import {
  useAddControlLayer,
  useAddGlobalReferenceImage,
  useAddInpaintMask,
  useAddRasterLayer,
  useAddRegionalGuidance,
  useAddRegionalReferenceImage,
} from 'features/controlLayers/hooks/addLayerHooks';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { selectIsFLUX } from 'features/controlLayers/store/paramsSlice';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiPlusBold } from 'react-icons/pi';

export const EntityListGlobalActionBarAddLayerMenu = memo(() => {
  const { t } = useTranslation();
  const isBusy = useCanvasIsBusy();
  const addGlobalReferenceImage = useAddGlobalReferenceImage();
  const addInpaintMask = useAddInpaintMask();
  const addRegionalGuidance = useAddRegionalGuidance();
  const addRegionalReferenceImage = useAddRegionalReferenceImage();
  const addRasterLayer = useAddRasterLayer();
  const addControlLayer = useAddControlLayer();
  const isFLUX = useAppSelector(selectIsFLUX);

  return (
    <Menu>
      <MenuButton
        as={IconButton}
        size="sm"
        variant="link"
        alignSelf="stretch"
        tooltip={t('controlLayers.addLayer')}
        aria-label={t('controlLayers.addLayer')}
        icon={<PiPlusBold />}
        data-testid="control-layers-add-layer-menu-button"
        isDisabled={isBusy}
      />
      <MenuList>
        <MenuGroup title={t('controlLayers.global')}>
          <MenuItem icon={<PiPlusBold />} onClick={addGlobalReferenceImage} isDisabled={isFLUX}>
            {t('controlLayers.globalReferenceImage')}
          </MenuItem>
        </MenuGroup>
        <MenuGroup title={t('controlLayers.regional')}>
          <MenuItem icon={<PiPlusBold />} onClick={addInpaintMask}>
            {t('controlLayers.inpaintMask')}
          </MenuItem>
          <MenuItem icon={<PiPlusBold />} onClick={addRegionalGuidance} isDisabled={isFLUX}>
            {t('controlLayers.regionalGuidance')}
          </MenuItem>
          <MenuItem icon={<PiPlusBold />} onClick={addRegionalReferenceImage} isDisabled={isFLUX}>
            {t('controlLayers.regionalReferenceImage')}
          </MenuItem>
        </MenuGroup>
        <MenuGroup title={t('controlLayers.layer_other')}>
          <MenuItem icon={<PiPlusBold />} onClick={addControlLayer}>
            {t('controlLayers.controlLayer')}
          </MenuItem>
          <MenuItem icon={<PiPlusBold />} onClick={addRasterLayer}>
            {t('controlLayers.rasterLayer')}
          </MenuItem>
        </MenuGroup>
      </MenuList>
    </Menu>
  );
});

EntityListGlobalActionBarAddLayerMenu.displayName = 'EntityListGlobalActionBarAddLayerMenu';
