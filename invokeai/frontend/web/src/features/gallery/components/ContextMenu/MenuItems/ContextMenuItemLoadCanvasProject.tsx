import { MenuItem } from '@invoke-ai/ui-library';
import { useLoadCanvasProjectFromServerWithDialog } from 'features/controlLayers/components/LoadCanvasProjectConfirmationAlertDialog';
import { useCanvasProjectDTOContext } from 'features/gallery/contexts/CanvasProjectDTOContext';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiFileArrowUpBold } from 'react-icons/pi';

export const ContextMenuItemLoadCanvasProject = memo(() => {
  const { t } = useTranslation();
  const projectDTO = useCanvasProjectDTOContext();
  const queueLoad = useLoadCanvasProjectFromServerWithDialog();

  const onClick = useCallback(() => {
    queueLoad(projectDTO.project_name);
  }, [queueLoad, projectDTO.project_name]);

  return (
    <MenuItem icon={<PiFileArrowUpBold />} onClickCapture={onClick}>
      {t('controlLayers.canvasProject.loadProject')}
    </MenuItem>
  );
});

ContextMenuItemLoadCanvasProject.displayName = 'ContextMenuItemLoadCanvasProject';
