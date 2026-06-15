import { IconMenuItem } from 'common/components/IconMenuItem';
import { useDownloadItem } from 'common/hooks/useDownloadImage';
import { useCanvasProjectDTOContext } from 'features/gallery/contexts/CanvasProjectDTOContext';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiDownloadSimpleBold } from 'react-icons/pi';

export const ContextMenuItemDownloadCanvasProject = memo(() => {
  const { t } = useTranslation();
  const projectDTO = useCanvasProjectDTOContext();
  const { downloadItem } = useDownloadItem();

  const onClick = useCallback(() => {
    // Suggest a friendlier filename than the bare UUID — append `.invk` so the OS associates it
    // with the canvas project loader on re-import.
    downloadItem(projectDTO.project_url, `${projectDTO.name}.invk`);
  }, [downloadItem, projectDTO]);

  return (
    <IconMenuItem
      icon={<PiDownloadSimpleBold />}
      aria-label={t('gallery.download')}
      tooltip={t('gallery.download')}
      onClick={onClick}
    />
  );
});

ContextMenuItemDownloadCanvasProject.displayName = 'ContextMenuItemDownloadCanvasProject';
