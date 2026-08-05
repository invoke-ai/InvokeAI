import { InvkFormatError } from './invk/format';

/**
 * Turn a failed project-file read into something worth showing.
 *
 * Every import entry point — Home, the Projects page, the in-editor Open dialog
 * — used to pass `error.message` straight into a toast, which meant a person who
 * picked the wrong file got a sentence written for a developer. `InvkFormatError`
 * carries a `reason` instead of a message so the wording lives in the catalog
 * and, importantly, says something different for each way this can go wrong:
 * the fix for a legacy canvas project is not the fix for a corrupt archive.
 *
 * Anything that is not an `InvkFormatError` reached us from the network or the
 * reducer. Those already carry messages meant for people, so they pass through.
 */
export const describeProjectFileError = (error: unknown, t: (key: string) => string): string | undefined => {
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
      return t('projects.file.damaged');
    }
    case 'too-large': {
      return t('projects.file.tooLarge');
    }
    default: {
      return t('projects.file.notAProject');
    }
  }
};
