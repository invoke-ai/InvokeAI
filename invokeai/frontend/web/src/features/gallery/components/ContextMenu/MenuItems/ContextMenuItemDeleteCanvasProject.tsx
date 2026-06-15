import { IconMenuItem } from 'common/components/IconMenuItem';
import { useCanvasProjectDTOContext } from 'features/gallery/contexts/CanvasProjectDTOContext';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiTrashSimpleBold } from 'react-icons/pi';
import { useDeleteCanvasProjectMutation } from 'services/api/endpoints/canvasProjects';

export const ContextMenuItemDeleteCanvasProject = memo(() => {
  const { t } = useTranslation();
  const projectDTO = useCanvasProjectDTOContext();
  const [deleteCanvasProject] = useDeleteCanvasProjectMutation();

  const onClick = useCallback(() => {
    // Mirror the video flow: one-step native confirm. Canvas projects don't carry references
    // into nodes/refs/canvas the way images do, so the heavier image usage modal isn't needed.
    if (window.confirm(t('gallery.deleteCanvasProjectConfirmation'))) {
      deleteCanvasProject({ project_name: projectDTO.project_name });
    }
  }, [deleteCanvasProject, projectDTO.project_name, t]);

  return (
    <IconMenuItem
      icon={<PiTrashSimpleBold />}
      onClickCapture={onClick}
      aria-label={t('gallery.deleteCanvasProject', { count: 1 })}
      tooltip={t('gallery.deleteCanvasProject', { count: 1 })}
      isDestructive
    />
  );
});

ContextMenuItemDeleteCanvasProject.displayName = 'ContextMenuItemDeleteCanvasProject';
