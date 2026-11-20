/* oxlint-disable react-perf/jsx-no-new-function-as-prop */
import type { GenerateModelConfig, GenerateSettings } from '@features/generation/core/types';

import { MenuActionItem } from '@platform/ui/Menu';
import { useTranslation } from 'react-i18next';

import { flushGenerateDrafts } from './generateDraftRegistry';
import { offerGenerateUndo } from './generateUndo';
import { useGenerateValues, useGenerationUi } from './GenerationUiContext';
import {
  describeChanges,
  describeMissingComponents,
  describeUnset,
  getApplyModelDefaultsPlan,
  getChangedSettings,
  getModelDefaultsPatch,
} from './shared/modelDefaultsReset';

export interface GenerateResetSelection {
  selectedModel: GenerateModelConfig;
  settings: GenerateSettings;
}

interface MenuItemState {
  hint: string | undefined;
  unavailableReason: string | undefined;
}

/** Mounted only while the menu shows (until its exit animation ends), so edits re-render the items only then. */
export const GenerateResetMenuItems = ({
  resolve,
}: {
  resolve: (values: Record<string, unknown>) => GenerateResetSelection | null;
}) => {
  const { i18n, t } = useTranslation();
  const locale = i18n.resolvedLanguage;
  const ui = useGenerationUi();
  const models = ui.models.catalog;
  const projectId = ui.project.activeProjectId;
  const items = useGenerateValues(
    (values): { apply: MenuItemState; reset: MenuItemState } | null => {
      const resolved = resolve(values);

      if (!resolved) {
        return null;
      }

      const { selectedModel, settings } = resolved;
      const name = selectedModel.name;
      const plan = getApplyModelDefaultsPlan(settings, selectedModel, models);
      const applyChanges = describeChanges(
        plan.changed,
        settings,
        { ...settings, ...plan.patch },
        selectedModel,
        t,
        locale
      );
      // One list of the defaults the model stores that cannot apply (not installed, or the VAE does not fit), so the
      // wording tells each once, whether or not the slot holds a pick.
      const missing = [...(plan.isVaeMissing ? (['vae'] as const) : []), ...plan.missingComponents];
      const missingList = describeMissingComponents(missing, selectedModel, { ...settings, ...plan.patch }, t, locale);
      const hasMissing = missing.length > 0;
      // A model that sets nothing leaves everything as it is; its reason says so without listing it all.
      const setsNothing = plan.defined.length === 0 && !hasMissing;
      const unset = setsNothing ? null : describeUnset(plan.unset, selectedModel, t, locale);
      const withMissingNote = (hint: string | undefined) => {
        if (!hasMissing) {
          return hint;
        }

        const note = t('widgets.generate.modelDefaultsMissingNote', { count: missing.length, list: missingList });

        return hint ? t('widgets.generate.modelDefaultsWithMissingNote', { hint, note }) : note;
      };
      const resetAfter = { ...settings, ...getModelDefaultsPatch(settings, selectedModel, models) };
      const resetChanged = getChangedSettings(settings, resetAfter, selectedModel);

      const applyUnavailableReason = setsNothing
        ? t('widgets.generate.modelDefaultsNoneSet', { name })
        : plan.changed.length > 0
          ? undefined
          : !hasMissing
            ? t('widgets.generate.modelDefaultsAlreadyApplied', { name })
            : plan.defined.length === 0
              ? t('widgets.generate.modelDefaultsOnlyMissing', { count: missing.length, list: missingList, name })
              : t('widgets.generate.modelDefaultsAppliedExceptMissing', {
                  count: missing.length,
                  list: missingList,
                  name,
                });
      const applyHint =
        applyChanges && unset
          ? t('widgets.generate.modelDefaultsApplyHint', { changes: applyChanges, settings: unset })
          : applyChanges || (unset ? t('widgets.generate.modelDefaultsUnset', { settings: unset }) : undefined);

      return {
        apply: {
          // An unavailable item's reason already names what is missing; the hint names it only once.
          hint: applyUnavailableReason ? applyHint : withMissingNote(applyHint),
          unavailableReason: applyUnavailableReason,
        },
        reset: {
          hint:
            resetChanged.length > 0
              ? withMissingNote(describeChanges(resetChanged, settings, resetAfter, selectedModel, t, locale))
              : t('widgets.generate.resetEverythingToDefaultsHint'),
          unavailableReason: resetChanged.length === 0 ? t('widgets.generate.resetEverythingAlreadyDone') : undefined,
        },
      };
    },
    (a, b) =>
      a === b ||
      (a !== null &&
        b !== null &&
        a.apply.hint === b.apply.hint &&
        a.apply.unavailableReason === b.apply.unavailableReason &&
        a.reset.hint === b.reset.hint &&
        a.reset.unavailableReason === b.reset.unavailableReason)
  );

  /** One write and one Undo for whatever the patch changes, its toast naming each change. */
  const commitReset = (
    getPatch: (settings: GenerateSettings, model: GenerateModelConfig) => Partial<GenerateSettings>,
    title: (model: GenerateModelConfig) => string
  ) => {
    // Flush before reading so the reset preserves pending prompt edits.
    flushGenerateDrafts();
    const resolved = resolve(ui.generateValues.getSnapshot());

    if (!resolved) {
      return;
    }

    const { selectedModel, settings } = resolved;
    const patch = getPatch(settings, selectedModel);
    const after = { ...settings, ...patch };
    const message = describeChanges(
      getChangedSettings(settings, after, selectedModel),
      settings,
      after,
      selectedModel,
      t,
      locale
    );

    ui.settings.patchGenerateSettings(patch, projectId);
    offerGenerateUndo({
      after,
      before: settings,
      ...(message ? { message } : {}),
      projectId,
      slot: 'settings',
      t,
      title: title(selectedModel),
      ui,
    });
  };

  const applyModelDefaults = () =>
    commitReset(
      (settings, model) => getApplyModelDefaultsPlan(settings, model, models).patch,
      (model) => t('widgets.generate.applyModelDefaultsDone', { name: model.name })
    );
  const resetEverything = () =>
    commitReset(
      (settings, model) => getModelDefaultsPatch(settings, model, models),
      () => t('widgets.generate.resetEverythingToDefaultsDone')
    );

  return (
    <>
      <MenuActionItem
        hint={items?.apply.hint}
        label={t('widgets.generate.applyModelDefaults')}
        unavailableReason={items?.apply.unavailableReason}
        value="apply-model-defaults"
        onSelect={applyModelDefaults}
      />
      <MenuActionItem
        hint={items?.reset.hint}
        label={t('widgets.generate.resetEverythingToDefaults')}
        unavailableReason={items?.reset.unavailableReason}
        value="reset-everything"
        onSelect={resetEverything}
      />
    </>
  );
};
