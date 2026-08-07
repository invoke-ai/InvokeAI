import { InvkFormatError } from './invk/format';
import { ProjectFlushError } from './projectFlush';

/**
 * Turn a failed project-file read or write into something worth showing. `InvkFormatError` carries
 * a `reason` rather than a message so the wording lives in the catalog and differs per failure: the
 * fix for a legacy canvas project is not the fix for a corrupt archive.
 *
 * `direction` matters for the two reasons both halves raise. Telling someone their project "is too
 * large to open" while they were exporting it names the wrong operation and the wrong file.
 *
 * `ProjectFlushError` is not a bad file but a project whose newest content never reached the
 * server. Anything else already carries a message meant for people and passes through.
 */
export type ProjectFileDirection = 'read' | 'write';

export const describeProjectFileError = (
  error: unknown,
  t: (key: string) => string,
  direction: ProjectFileDirection = 'read'
): string | undefined => {
  if (error instanceof ProjectFlushError) {
    return t(error.reason === 'unsynced' ? 'projects.file.notSynced' : 'projects.file.supersededElsewhere');
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
