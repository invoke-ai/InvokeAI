import { IconButton, Menu, MenuButton, MenuList } from '@invoke-ai/ui-library';
import { CanvasEntityListMenuItems } from 'features/controlLayers/components/CanvasEntityList/CanvasEntityListMenuItems';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiDotsThreeOutlineFill } from 'react-icons/pi';

export const CanvasEntityListMenuButton = memo(() => {
  const { t } = useTranslation();

  return (
    <Menu>
      <MenuButton
        as={IconButton}
        aria-label={t('accessibility.menu')}
        icon={<PiDotsThreeOutlineFill />}
        variant="link"
        data-testid="control-layers-add-layer-menu-button"
        alignSelf="stretch"
      />
      <MenuList>
        <CanvasEntityListMenuItems />
      </MenuList>
    </Menu>
  );
});

CanvasEntityListMenuButton.displayName = 'CanvasEntityListMenuButton';
