import type { RefusedWorkbenchProject } from '@workbench/projectContracts';
import type { TFunction } from 'i18next';

import { MIN_SUPPORTED_CANVAS_SCHEMA_VERSION } from '@workbench/canvasSchemaVersion';

export interface RefusedProjectNotice {
  title: string;
  message: string;
}

/** The user-facing explanation for one project the canvas version gate refused. */
export const describeRefusedProject = (refused: RefusedWorkbenchProject, t: TFunction): RefusedProjectNotice => {
  const name = refused.projectName;

  return {
    message:
      refused.source === 'project-document'
        ? t(refused.refusal.status === 'malformed' ? 'projects.load.malformed' : 'projects.load.unsupportedVersion', {
            name,
          })
        : refused.refusal.status === 'unsupported-version'
          ? t(
              refused.refusal.version < MIN_SUPPORTED_CANVAS_SCHEMA_VERSION
                ? 'projects.load.legacyVersion'
                : 'projects.load.unsupportedVersion',
              { name }
            )
          : t('projects.load.invalid', { name }),
    title: t('projects.couldNotOpen'),
  };
};

/** One notice for everything refused while hydrating a session. */
export const describeRefusedProjects = (
  refused: readonly RefusedWorkbenchProject[],
  t: TFunction
): RefusedProjectNotice | null => {
  if (refused.length === 0) {
    return null;
  }
  if (refused.length === 1) {
    return describeRefusedProject(refused[0]!, t);
  }

  return {
    message: t('projects.load.multiple', {
      count: refused.length,
      names: refused.map((r) => r.projectName).join(', '),
    }),
    title: t('projects.couldNotOpen'),
  };
};
