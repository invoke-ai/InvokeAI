import type { TFunction } from 'i18next';

import { getGenerateValuesRevertPatch } from '@features/generation/core/settings';
import { areJsonValuesStructurallyEqual } from '@platform/core/json';

import type { GenerationUiAdapter } from './GenerationUiContext';

import { flushGenerateDrafts } from './generateDraftRegistry';

/**
 * Generate's Undo slots, one pair per project. A newer Undo in a slot withdraws the older, since undoing an older step
 * after a newer one would leave a mix of both. Whole-form changes (a model switch, a preset, Reset all) share one slot;
 * prompt changes that Ctrl+Z in the prompt cannot undo (a template choice, a history entry, a rewrite) share the other.
 */
export const getGenerateSettingsUndoKey = (projectId: string): string => `generate-settings:${projectId}`;
export const getGeneratePromptUndoKey = (projectId: string): string => `generate-prompt:${projectId}`;

interface GenerateUndoOffer {
  /** The project's stored values after the change. */
  after: object;
  /** The project's stored values before the change. */
  before: object;
  message?: string;
  projectId: string;
  slot: 'prompt' | 'settings';
  t: TFunction;
  title: string;
  ui: Pick<GenerationUiAdapter, 'notifications' | 'settings'>;
  /**
   * The change only makes sense whole (a template merged into the prompt): once any key it set has changed since, the
   * Undo does nothing. Otherwise it puts back the keys still holding what the change wrote and leaves the rest.
   */
  isWhole?: boolean;
}

/**
 * The entries of `revertPatch` whose key still holds what the change wrote, so restoring them never discards a later
 * edit; null when the project is gone. Typing still inside a draft's debounce counts as an edit since, so drafts flush
 * before the comparison.
 */
export const readStillAppliedRevert = (
  settings: Pick<GenerationUiAdapter['settings'], 'readGenerateValues'>,
  projectId: string,
  revertPatch: Readonly<Record<string, unknown>>,
  written: object
): [string, unknown][] | null => {
  flushGenerateDrafts();
  const current = settings.readGenerateValues(projectId);
  const writtenValues = written as Readonly<Record<string, unknown>>;

  return current
    ? Object.entries(revertPatch).filter(([key]) => areJsonValuesStructurallyEqual(current[key], writtenValues[key]))
    : null;
};

/**
 * Reports a Generate change that already happened, with an Undo that restores the earlier value of each key the
 * change set. A key edited since keeps the edit, so an Undo never discards later work; with nothing left to restore it
 * says so. No toast when the change set nothing.
 */
export const offerGenerateUndo = ({
  after,
  before,
  isWhole = false,
  message,
  projectId,
  slot,
  t,
  title,
  ui,
}: GenerateUndoOffer): void => {
  const revertPatch = getGenerateValuesRevertPatch(before, after);

  if (Object.keys(revertPatch).length === 0) {
    return;
  }

  ui.notifications.undoable(title, {
    ...(message ? { message } : {}),
    onUndo: () => {
      const stillApplied = readStillAppliedRevert(ui.settings, projectId, revertPatch, after);

      if (
        stillApplied &&
        stillApplied.length > 0 &&
        (!isWhole || stillApplied.length === Object.keys(revertPatch).length)
      ) {
        ui.settings.patchGenerateSettings(Object.fromEntries(stillApplied), projectId);
      } else {
        ui.notifications.info(t('widgets.generate.undoStale'));
      }
    },
    projectId,
    supersedes: slot === 'prompt' ? getGeneratePromptUndoKey(projectId) : getGenerateSettingsUndoKey(projectId),
  });
};
