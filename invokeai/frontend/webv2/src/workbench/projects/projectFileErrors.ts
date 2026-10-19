import { InvkFormatError } from './invk/format';
import { ProjectFlushError } from './projectFlush';

/** Localize failure reason and direction; flush errors are distinct from archive corruption. */
export type ProjectFileDirection = 'read' | 'write';

/**
 * An export asked to leave the workspace out of a project held whole that this build cannot read: only a document it
 * can read can be split, and the verbatim recovery copy would carry the workspace.
 */
export class ProjectWorkspaceInseparableError extends Error {
  constructor() {
    super('This project cannot be read here, so its workspace cannot be left out of the file.');
    this.name = 'ProjectWorkspaceInseparableError';
  }
}

export const describeProjectFileError = (
  error: unknown,
  t: (key: string) => string,
  direction: ProjectFileDirection = 'read'
): string | undefined => {
  if (error instanceof ProjectFlushError) {
    return t(
      error.reason === 'schema-refused'
        ? 'projects.file.updateClient'
        : error.reason === 'unsynced'
          ? 'projects.file.notSynced'
          : 'projects.file.supersededElsewhere'
    );
  }

  if (error instanceof ProjectWorkspaceInseparableError) {
    return t('projects.file.workspaceInseparable');
  }

  if (!(error instanceof InvkFormatError)) {
    return error instanceof Error ? error.message : undefined;
  }

  switch (error.reason) {
    case 'legacy-canvas-project': {
      return t('projects.file.legacyCanvasProject');
    }
    case 'unsupported-version': {
      return t('projects.file.unsupportedVersion');
    }
    case 'damaged': {
      return t(direction === 'write' ? 'projects.file.damagedProject' : 'projects.file.damaged');
    }
    case 'too-large': {
      return t(direction === 'write' ? 'projects.file.tooLargeToWrite' : 'projects.file.tooLarge');
    }
    default: {
      return t('projects.file.notAProject');
    }
  }
};
