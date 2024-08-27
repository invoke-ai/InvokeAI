import { IconButton, Menu, MenuButton, MenuList } from '@invoke-ai/ui-library';
import { CanvasEntityListMenuItems } from 'features/controlLayers/components/CanvasEntityList/EntityListActionBarAddLayerMenuItems';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiPlusBold } from 'react-icons/pi';

export const EntityListActionBarAddLayerButton = memo(() => {
  const { t } = useTranslation();

  return (
    <Menu>
      <MenuButton
        as={IconButton}
        size="sm"
        tooltip={t('controlLayers.addLayer')}
        aria-label={t('controlLayers.addLayer')}
        icon={<PiPlusBold />}
        variant="ghost"
        data-testid="control-layers-add-layer-menu-button"
      />
      <MenuList>
        <CanvasEntityListMenuItems />
      </MenuList>
    </Menu>
  );
});

EntityListActionBarAddLayerButton.displayName = 'EntityListActionBarAddLayerButton';
