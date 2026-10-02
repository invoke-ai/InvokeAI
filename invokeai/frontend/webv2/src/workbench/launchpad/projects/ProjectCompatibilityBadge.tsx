import type { ProjectSummary } from '@workbench/projects/library';

import { Badge } from '@chakra-ui/react';
import { MIN_SUPPORTED_CANVAS_SCHEMA_VERSION } from '@workbench/canvasSchemaVersion';
import { isProjectSummaryCompatible } from '@workbench/projects/library';
import { useTranslation } from 'react-i18next';

/** A stable library-level cue; opening the editor is too late to explain incompatibility. */
export const ProjectCompatibilityBadge = ({ summary }: { summary: ProjectSummary }) => {
  const { t } = useTranslation();

  if (isProjectSummaryCompatible(summary)) {
    return null;
  }

  return (
    <Badge alignSelf="flex-start" colorPalette="orange" size="xs" variant="surface">
      {t(
        summary.minimumCanvasSchemaVersion < MIN_SUPPORTED_CANVAS_SCHEMA_VERSION
          ? 'projects.legacyFormat'
          : 'projects.requiresNewerInvoke'
      )}
    </Badge>
  );
};
