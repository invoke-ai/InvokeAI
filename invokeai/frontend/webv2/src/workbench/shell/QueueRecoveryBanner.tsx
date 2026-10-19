import { useProjectActions } from '@workbench/projects/useProjectActions';
import { QueueRecoveryNotice } from '@workbench/queue-integration/QueueRecoveryNotice';
import { shallowEqual, useWorkbenchSelector } from '@workbench/WorkbenchContext';
import { useCallback } from 'react';

export const QueueRecoveryBanner = () => {
  const openProjectIds = useWorkbenchSelector(
    (snapshot) => snapshot.projects.map((project) => project.id),
    shallowEqual
  );
  const { openProject } = useProjectActions();
  const handleOpen = useCallback((projectId: string) => openProject(projectId, projectId), [openProject]);

  return <QueueRecoveryNotice openProjectIds={openProjectIds} onOpen={handleOpen} />;
};
