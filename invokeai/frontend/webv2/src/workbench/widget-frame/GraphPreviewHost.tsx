import type { GraphBearingSurfaceContract } from '@workbench/widgetContracts';

import { ensureModelsLoaded, useModelsSelector } from '@features/models';
import { GraphPreviewDialog } from '@features/workflow/preview';
import { ensureInvocationTemplatesLoaded, useInvocationTemplatesSnapshot } from '@features/workflow/react';
import { useMountEffect } from '@platform/react/useMountEffect';
import { useActiveProjectSelector } from '@workbench/WorkbenchContext';
import { useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { buildGraphPreviewSource } from './graphPreviewSource';

/**
 * Mounted only while the preview dialog is open (it subscribes to the whole
 * active project so the compiled graph tracks every settings edit live).
 */
export const GraphPreviewHost = ({
  isOpen,
  surface,
  onOpenChange,
}: {
  isOpen: boolean;
  surface: GraphBearingSurfaceContract;
  onOpenChange: (isOpen: boolean) => void;
}) => {
  const { t } = useTranslation();
  const project = useActiveProjectSelector((activeProject) => activeProject);
  const modelsStatus = useModelsSelector((snapshot) => snapshot.status);
  const models = useModelsSelector((snapshot) => snapshot.models);
  const templates = useInvocationTemplatesSnapshot();
  const availabilityModels = modelsStatus === 'loaded' ? models : undefined;
  const source = useMemo(
    () => buildGraphPreviewSource({ models: availabilityModels, project, surface, t, templates }),
    [availabilityModels, project, surface, t, templates]
  );

  useMountEffect(() => {
    void ensureModelsLoaded();
    ensureInvocationTemplatesLoaded();
  });

  return (
    <GraphPreviewDialog
      graph={source.graph}
      graphId={surface.graphId}
      isOpen={isOpen}
      positionHints={source.positionHints}
      sourceId={surface.sourceId}
      title={surface.label}
      onOpenChange={onOpenChange}
    />
  );
};

export default GraphPreviewHost;
