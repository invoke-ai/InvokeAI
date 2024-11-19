import { IconButton, Menu, MenuButton, MenuItem, MenuList } from '@invoke-ai/ui-library';
import {
  useNewCanvasSession,
  useNewGallerySession,
} from 'features/controlLayers/components/NewSessionConfirmationAlertDialog';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiFilePlusBold, PiImageBold, PiPaintBrushBold } from 'react-icons/pi';

export const CanvasToolbarNewSessionMenuButton = memo(() => {
  const { t } = useTranslation();
  const { newGallerySessionWithDialog } = useNewGallerySession();
  const { newCanvasSessionWithDialog } = useNewCanvasSession();
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
        <MenuItem icon={<PiImageBold />} onClick={newGallerySessionWithDialog}>
          {t('controlLayers.newGallerySession')}
        </MenuItem>
        <MenuItem icon={<PiPaintBrushBold />} onClick={newCanvasSessionWithDialog}>
          {t('controlLayers.newCanvasSession')}
        </MenuItem>
      </MenuList>
    </Menu>
  );
});

CanvasToolbarNewSessionMenuButton.displayName = 'CanvasToolbarNewSessionMenuButton';
