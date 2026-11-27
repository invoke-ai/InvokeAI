import { IconButton, Menu, MenuButton, MenuList } from '@invoke-ai/ui-library';
import { SessionMenuItems } from 'common/components/SessionMenuItems';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiFilePlusBold } from 'react-icons/pi';

export const CanvasToolbarNewSessionMenuButton = memo(() => {
  const { t } = useTranslation();
  return (
    <Menu placement="bottom-end">
      <MenuButton
        as={IconButton}
        aria-label={t('controlLayers.newSession')}
        icon={<PiFilePlusBold />}
        variant="link"
        alignSelf="stretch"
      />
      <MenuList>
        <SessionMenuItems />
      </MenuList>
    </Menu>
  );
});

CanvasToolbarNewSessionMenuButton.displayName = 'CanvasToolbarNewSessionMenuButton';
