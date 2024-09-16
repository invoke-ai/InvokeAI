import { IconButton, Menu, MenuButton, MenuGroup, MenuItem, MenuList } from '@invoke-ai/ui-library';
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

export const EntityListGlobalActionBarAddLayerMenu = memo(() => {
  const { t } = useTranslation();
  const addInpaintMask = useAddInpaintMask();
  const addRegionalGuidance = useAddRegionalGuidance();
  const addRasterLayer = useAddRasterLayer();
  const addControlLayer = useAddControlLayer();
  const addIPAdapter = useAddIPAdapter();

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
      />
      <MenuList>
        <MenuGroup title={t('controlLayers.global')}>
          <MenuItem icon={<PiPlusBold />} onClick={addIPAdapter}>
            {t('controlLayers.ipAdapter')}
          </MenuItem>
        </MenuGroup>
        <MenuGroup title={t('controlLayers.regional')}>
          <MenuItem icon={<PiPlusBold />} onClick={addInpaintMask}>
            {t('controlLayers.inpaintMask')}
          </MenuItem>
          <MenuItem icon={<PiPlusBold />} onClick={addRegionalGuidance}>
            {t('controlLayers.regionalGuidance')}
          </MenuItem>
        </MenuGroup>
        <MenuGroup title={t('controlLayers.layer_other')}>
          <MenuItem icon={<PiPlusBold />} onClick={addRasterLayer}>
            {t('controlLayers.rasterLayer')}
          </MenuItem>
          <MenuItem icon={<PiPlusBold />} onClick={addControlLayer}>
            {t('controlLayers.controlLayer')}
          </MenuItem>
        </MenuGroup>
      </MenuList>
    </Menu>
  );
});

EntityListGlobalActionBarAddLayerMenu.displayName = 'EntityListGlobalActionBarAddLayerMenu';
