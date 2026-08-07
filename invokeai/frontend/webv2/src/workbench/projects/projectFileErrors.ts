import { InvkFormatError } from './invk/format';
import { ProjectFlushError } from './projectFlush';

/**
 * Turn a failed project-file read or write into something worth showing.
 *
 * Every import entry point — Home, the Projects page, the in-editor Open dialog
 * — used to pass `error.message` straight into a toast, which meant a person who
 * picked the wrong file got a sentence written for a developer. `InvkFormatError`
 * carries a `reason` instead of a message so the wording lives in the catalog
 * and, importantly, says something different for each way this can go wrong:
 * the fix for a legacy canvas project is not the fix for a corrupt archive.
 *
 * The direction matters for the two reasons both halves can raise. `too-large`
 * comes from the export planner, the archive writer and the response reader as
 * readily as from opening a file, and `damaged` is raised by duplication for a
 * document that will not rehydrate. Telling someone their project "is too large
 * to open" while they were exporting it names the wrong operation and the wrong
 * file — theirs, not one they were handed.
 *
 * `ProjectFlushError` is the third case: not a bad file but a project whose
 * newest content never reached the server, so there is nothing correct to
 * export or copy yet. Its own message names the mechanism rather than the fix,
 * which is the wrong half to show someone.
 *
 * Anything else reached us from the network or the reducer. Those already carry
 * messages meant for people, so they pass through.
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
