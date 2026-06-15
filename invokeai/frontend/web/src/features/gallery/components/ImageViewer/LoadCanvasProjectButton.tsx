import { Button } from '@invoke-ai/ui-library';
import { useLoadCanvasProjectFromServerWithDialog } from 'features/controlLayers/components/LoadCanvasProjectConfirmationAlertDialog';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiFileArrowUpBold } from 'react-icons/pi';
import type { CanvasProjectDTO } from 'services/api/types';

type Props = {
  projectDTO: CanvasProjectDTO;
};

/**
 * Toolbar button that triggers the load-confirmation dialog with the selected canvas project as
 * source. The actual ZIP fetch + state restore happens after the user confirms.
 */
export const LoadCanvasProjectButton = memo(({ projectDTO }: Props) => {
  const { t } = useTranslation();
  const queueLoad = useLoadCanvasProjectFromServerWithDialog();
  const onClick = useCallback(() => {
    queueLoad(projectDTO.project_name);
  }, [queueLoad, projectDTO.project_name]);

  return (
    <Button size="sm" colorScheme="invokeBlue" leftIcon={<PiFileArrowUpBold />} onClick={onClick}>
      {t('controlLayers.canvasProject.loadProject')}
    </Button>
  );
});

LoadCanvasProjectButton.displayName = 'LoadCanvasProjectButton';
