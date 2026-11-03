import type {
  GenerateModelSelectionResult,
  GenerateModelSwitchChange,
  GenerateModelSwitchChangeKind,
} from '@features/generation/core/baseGenerationPolicies';
import type { GenerateModelConfig } from '@features/generation/core/types';
import type { TFunction } from 'i18next';

import { registerAccountOwnedResource } from '@platform/state/accountLifecycle';
import { createExternalStore } from '@platform/state/externalStore';

import type { GenerationUiAdapter } from './GenerationUiContext';

import { getGenerateSettingsUndoKey, readStillAppliedRevert } from './generateUndo';

/** The order a switch's changes are told in: what is lost first, then what is only set aside or moved. */
const MODEL_SWITCH_CHANGE_KINDS: readonly GenerateModelSwitchChangeKind[] = [
  'removed',
  'reset',
  'turnedOff',
  'adjusted',
];

/**
 * One sentence per kind of change, then the size the change moved to; empty when nothing else changed. The toast says
 * what happened (`switchChanges`), the confirmation what would happen (`switchModelLines`).
 */
export const describeModelSwitchChanges = (
  result: Pick<GenerateModelSelectionResult, 'changes' | 'dimensionsAdjusted' | 'settings'>,
  t: TFunction,
  locale: string | undefined,
  keyPrefix: 'widgets.generate.switchChanges' | 'widgets.generate.switchModelLines' = 'widgets.generate.switchChanges'
): string[] => {
  const formatList = (changes: readonly GenerateModelSwitchChange[]) =>
    new Intl.ListFormat(locale, { style: 'long', type: 'conjunction' }).format(
      changes.map((change) =>
        t(
          `widgets.generate.switchSubjects.${change.subject}`,
          change.count === undefined ? {} : { count: change.count }
        )
      )
    );
  const sentences = MODEL_SWITCH_CHANGE_KINDS.flatMap((kind) => {
    const changes = result.changes.filter((change) => change.kind === kind);

    return changes.length > 0 ? [t(`${keyPrefix}.${kind}`, { labels: formatList(changes) })] : [];
  });

  return result.dimensionsAdjusted
    ? [
        ...sentences,
        t('widgets.generate.switchChanges.size', { height: result.settings.height, width: result.settings.width }),
      ]
    : sentences;
};

interface ModelSwitchNotifier {
  info(title: string, message?: string): void;
  undoable(
    title: string,
    notice: { message?: string; onUndo: () => void; projectId?: string; supersedes?: string }
  ): void;
}

/**
 * Reports a switch that already happened, naming what it set aside, with an Undo that restores the earlier choices.
 * The Undo acts only while the project still uses the model the switch picked, and then puts back only the keys that
 * still hold what the switch wrote, so an edit made since (steps, size) is kept; a newer whole-form change withdraws it.
 * Call it right after the switch committed synchronously: what the project stores then is taken as what the switch
 * wrote, which the commit resolved and so may differ from `result.settings`.
 */
export const notifyGenerateModelSwitched = ({
  locale,
  model,
  notifications,
  projectId,
  result,
  revertPatch,
  settings,
  t,
}: {
  locale: string | undefined;
  model: Pick<GenerateModelConfig, 'key' | 'name'>;
  notifications: ModelSwitchNotifier;
  projectId: string;
  result: Pick<GenerateModelSelectionResult, 'changes' | 'dimensionsAdjusted' | 'settings'>;
  /** The earlier value of each key the switch changed. */
  revertPatch: Readonly<Record<string, unknown>>;
  settings: Pick<GenerationUiAdapter['settings'], 'patchGenerateSettings' | 'readGenerateValues'>;
  t: TFunction;
}): void => {
  const written = settings.readGenerateValues(projectId) ?? {};
  const message = describeModelSwitchChanges(result, t, locale).join(' ');

  notifications.undoable(t('widgets.generate.switchedModel', { name: model.name }), {
    ...(message ? { message } : {}),
    onUndo: () => {
      if (settings.readGenerateValues(projectId)?.modelKey !== model.key) {
        notifications.info(t('widgets.generate.switchUndoStale', { name: model.name }));
        return;
      }

      const stillApplied = readStillAppliedRevert(settings, projectId, revertPatch, written);

      if (stillApplied && stillApplied.length > 0) {
        settings.patchGenerateSettings(Object.fromEntries(stillApplied), projectId);
      } else {
        notifications.info(t('widgets.generate.undoStale'));
      }
    },
    projectId,
    supersedes: getGenerateSettingsUndoKey(projectId),
  });
};

/**
 * A switch waiting for the user to confirm what it loses. Any surface may ask (the palette as well as the model
 * picker); the project's model card shows the dialog and runs the switch, so every switch confirms and undoes alike.
 * One slot: a newer request replaces an older one, and it is dropped when the active project changes (the App clears
 * it) or the account ends, so a request never surfaces later in another context.
 */
const pendingSwitch = createExternalStore<{ request: { model: GenerateModelConfig; projectId: string } | null }>({
  request: null,
});

registerAccountOwnedResource({
  clear: () => pendingSwitch.setSnapshot({ request: null }),
  name: 'generate-model-switch-confirmation',
});

export const requestGenerateModelSwitchConfirmation = (request: {
  model: GenerateModelConfig;
  projectId: string;
}): void => pendingSwitch.setSnapshot({ request });

export const clearGenerateModelSwitchConfirmation = (): void => {
  if (pendingSwitch.getSnapshot().request) {
    pendingSwitch.setSnapshot({ request: null });
  }
};

/** The model awaiting confirmation for `projectId`, or null. */
export const usePendingGenerateModelSwitch = (projectId: string): GenerateModelConfig | null =>
  pendingSwitch.useSelector(({ request }) => (request?.projectId === projectId ? request.model : null), Object.is);
