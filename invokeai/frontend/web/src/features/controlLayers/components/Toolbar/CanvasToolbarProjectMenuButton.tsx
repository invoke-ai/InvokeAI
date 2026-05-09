import { IconButton, Menu, MenuButton, MenuItem, MenuList } from '@invoke-ai/ui-library';
import { useLoadCanvasProjectWithDialog } from 'features/controlLayers/components/LoadCanvasProjectConfirmationAlertDialog';
import { useSaveCanvasProjectWithDialog } from 'features/controlLayers/components/SaveCanvasProjectDialog';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArchiveBold, PiFileArrowDownBold, PiFileArrowUpBold } from 'react-icons/pi';

export const CanvasToolbarProjectMenuButton = memo(() => {
  const { t } = useTranslation();
  const isBusy = useCanvasIsBusy();
  const saveCanvasProject = useSaveCanvasProjectWithDialog();
  const loadCanvasProject = useLoadCanvasProjectWithDialog();

  return (
    <Menu placement="bottom-end">
      <MenuButton
        as={IconButton}
        aria-label={t('controlLayers.canvasProject.project')}
        tooltip={t('controlLayers.canvasProject.project')}
        icon={<PiArchiveBold />}
        variant="link"
        alignSelf="stretch"
      />
      <MenuList>
        <MenuItem icon={<PiFileArrowDownBold />} isDisabled={isBusy} onClick={saveCanvasProject}>
          {t('controlLayers.canvasProject.saveProject')}
        </MenuItem>
        <MenuItem icon={<PiFileArrowUpBold />} isDisabled={isBusy} onClick={loadCanvasProject}>
          {t('controlLayers.canvasProject.loadProject')}
        </MenuItem>
      </MenuList>
    </Menu>
  );
});

CanvasToolbarProjectMenuButton.displayName = 'CanvasToolbarProjectMenuButton';
