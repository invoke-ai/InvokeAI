/* oxlint-disable react-perf/jsx-no-new-function-as-prop */
import type { GenerationModelCatalogItem as ModelConfig } from '@features/generation/contracts';
import type { GenerateModelConfig, GenerateSettings, VaeModelConfig } from '@features/generation/core/types';

import { MenuActionItem } from '@platform/ui/Menu';
import { useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { flushGenerateDrafts } from './generateDraftRegistry';
import { offerGenerateUndo } from './generateUndo';
import { useGenerateValues, useGenerationUi } from './GenerationUiContext';
import {
  describeChanges,
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

/**
 * The reset menu's items. Mounted only while the menu shows, until its exit animation ends, so edits re-render them
 * only meanwhile.
 */
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
  const vaeModels = useMemo(
    () => models.filter((model): model is ModelConfig & VaeModelConfig => model.type === 'vae'),
    [models]
  );
  const items = useGenerateValues(
    (values): { apply: MenuItemState; reset: MenuItemState } | null => {
      const resolved = resolve(values);

      if (!resolved) {
        return null;
      }

      const { selectedModel, settings } = resolved;
      const name = selectedModel.name;
      const plan = getApplyModelDefaultsPlan(settings, selectedModel, vaeModels);
      const applyChanges = describeChanges(
        plan.changed,
        settings,
        { ...settings, ...plan.patch },
        selectedModel,
        t,
        locale
      );
      // A model that sets nothing leaves everything as it is; its reason says so without listing it all.
      const setsNothing = plan.defined.length === 0 && !plan.isVaeMissing;
      const unset = setsNothing ? null : describeUnset(plan.unset, selectedModel, t, locale);
      // A VAE the model sets but that cannot apply is no unset setting: it gets its own sentence.
      const withVaeNote = (hint: string | undefined) =>
        !plan.isVaeMissing
          ? hint
          : hint
            ? t('widgets.generate.modelDefaultsWithVaeMissing', { hint })
            : t('widgets.generate.modelDefaultsVaeMissing');
      const resetAfter = { ...settings, ...getModelDefaultsPatch(settings, selectedModel, vaeModels) };
      const resetChanged = getChangedSettings(settings, resetAfter, selectedModel);

      const applyUnavailableReason = setsNothing
        ? t('widgets.generate.modelDefaultsNoneSet', { name })
        : plan.changed.length > 0
          ? undefined
          : !plan.isVaeMissing
            ? t('widgets.generate.modelDefaultsAlreadyApplied', { name })
            : plan.defined.length === 0
              ? t('widgets.generate.modelDefaultsOnlyVaeMissing', { name })
              : t('widgets.generate.modelDefaultsAppliedExceptVae', { name });
      const applyHint =
        applyChanges && unset
          ? t('widgets.generate.modelDefaultsApplyHint', { changes: applyChanges, settings: unset })
          : applyChanges || (unset ? t('widgets.generate.modelDefaultsUnset', { settings: unset }) : undefined);

      return {
        apply: {
          // An unavailable item's reason already tells a missing VAE; the hint tells it only once.
          hint: applyUnavailableReason ? applyHint : withVaeNote(applyHint),
          unavailableReason: applyUnavailableReason,
        },
        reset: {
          hint:
            resetChanged.length > 0
              ? withVaeNote(describeChanges(resetChanged, settings, resetAfter, selectedModel, t, locale))
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
      (settings, model) => getApplyModelDefaultsPlan(settings, model, vaeModels).patch,
      (model) => t('widgets.generate.applyModelDefaultsDone', { name: model.name })
    );
  const resetEverything = () =>
    commitReset(
      (settings, model) => getModelDefaultsPatch(settings, model, vaeModels),
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
