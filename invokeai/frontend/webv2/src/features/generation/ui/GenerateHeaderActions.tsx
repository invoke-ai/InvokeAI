/* oxlint-disable react-perf/jsx-no-new-function-as-prop */
import type { GenerationModelCatalogItem as ModelConfig } from '@features/generation/contracts';
import type { VaeModelConfig } from '@features/generation/core/types';
import type { MouseEvent } from 'react';

import { Icon, Menu, Portal, Text } from '@chakra-ui/react';
import {
  getGenerateModelSelectionResult,
  isSupportedGenerateModel,
} from '@features/generation/core/baseGenerationPolicies';
import { normalizeGenerateSettings } from '@features/generation/core/settings';
import { resolveGenerateWidgetValues } from '@features/generation/settings';
import { IconButton, RenameDialog, Tooltip } from '@platform/ui';
import { MenuContent } from '@platform/ui/Menu';
import { BookmarkIcon, RotateCcwIcon, Trash2Icon } from 'lucide-react';
import { useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

import type { GeneratePresetRecord } from './GenerationUiContext';

import { flushGenerateDrafts } from './generateDraftRegistry';
import { getGenerateFormCommitPatch } from './generateFormViewModel';
import { useGenerationUi } from './GenerationUiContext';
import { notifyGenerateModelSelectionCleared } from './modelSelectionNotice';
import {
  getModelDefaultsPatch,
  getModelDefaultSettings,
  settingsMatchModelDefaults,
} from './shared/modelDefaultSettings';

/**
 * Widget-header actions: the preset library (save/apply/delete named settings
 * snapshots) and reset-every-model-governed-setting-to-model-defaults. They sit
 * in the header because they act on the whole panel, not one zone of it.
 */
export const GenerateHeaderActions = () => {
  const { i18n, t } = useTranslation();
  const ui = useGenerationUi();
  const [isSavingPreset, setIsSavingPreset] = useState(false);
  const models = ui.models.catalog;
  const projectId = ui.project.activeProjectId;
  const settings = normalizeGenerateSettings(ui.project.generateValues);
  const supportedModels = useMemo(() => models.filter(isSupportedGenerateModel), [models]);
  const selectedModel = useMemo(
    () => supportedModels.find((model) => model.key === settings?.modelKey),
    [supportedModels, settings?.modelKey]
  );
  const vaeModels = useMemo(
    () => models.filter((model): model is ModelConfig & VaeModelConfig => model.type === 'vae'),
    [models]
  );
  const modelDefaultSettings =
    settings && selectedModel ? getModelDefaultSettings(settings, selectedModel, vaeModels) : null;
  const isAtModelDefaults =
    settings && modelDefaultSettings ? settingsMatchModelDefaults(settings, modelDefaultSettings) : false;

  const resetToModelDefaults = () => {
    if (!selectedModel || !settings) {
      return;
    }

    // Flush pending debounced edits first so this patch lands on top of them;
    // the patch only carries model-governed keys, so flushed prompt edits survive.
    flushGenerateDrafts();
    ui.settings.patchGenerateSettings(getModelDefaultsPatch(settings, selectedModel, vaeModels), projectId);
  };

  const savePreset = (label: string) => {
    // Flush so the snapshot carries what the user sees, not a debounce behind it.
    flushGenerateDrafts();
    const snapshot = normalizeGenerateSettings(ui.project.generateValues);

    if (snapshot) {
      ui.presets.save(label, { ...snapshot });
    }
  };

  const applyPreset = (record: GeneratePresetRecord) => {
    const normalized = normalizeGenerateSettings(record.values);

    if (!normalized) {
      ui.notifications.error(t('widgets.generate.presetUnreadable'), record.label);
      return;
    }

    const model = supportedModels.find((candidate) => candidate.key === normalized.modelKey);

    if (!model) {
      ui.notifications.error(t('widgets.generate.presetModelMissing'), record.label);
      return;
    }

    // Reconcile against the current catalog (models installed or removed since
    // the preset was saved), then commit through the same resolve + patch path
    // the widget's own model selection takes.
    const result = getGenerateModelSelectionResult({ currentValues: normalized, model, models });
    const resolved = resolveGenerateWidgetValues({ models, storedValues: { ...result.settings, model } });

    if (!resolved) {
      ui.notifications.error(t('widgets.generate.presetUnreadable'), record.label);
      return;
    }

    ui.settings.patchGenerateSettings(getGenerateFormCommitPatch(resolved.values), projectId);
    notifyGenerateModelSelectionCleared({
      clearedLabels: result.clearedLabels,
      locale: i18n.resolvedLanguage,
      modelName: model.name,
      notifications: ui.notifications,
      t,
    });
  };

  const removePreset = (event: MouseEvent, presetId: string) => {
    // The row's click applies the preset; the trash icon only deletes it.
    event.stopPropagation();
    ui.presets.remove(presetId);
  };

  const presetsLabel = t('widgets.generate.presets');
  const resetLabel = t('widgets.generate.resetAllToModelDefaults');

  return (
    <>
      <Menu.Root>
        <Tooltip content={presetsLabel}>
          <Menu.Trigger asChild>
            <IconButton aria-label={presetsLabel} color="fg.muted" size="2xs" variant="ghost">
              <Icon as={BookmarkIcon} boxSize="3.5" />
            </IconButton>
          </Menu.Trigger>
        </Tooltip>
        <Portal>
          <Menu.Positioner>
            <MenuContent>
              <Menu.Item disabled={!selectedModel} value="save-preset" onClick={() => setIsSavingPreset(true)}>
                {t('widgets.generate.savePreset')}
              </Menu.Item>
              {ui.presets.presets.length > 0 ? <Menu.Separator /> : null}
              {ui.presets.presets.map((preset) => (
                <Menu.Item key={preset.id} value={preset.id} onClick={() => applyPreset(preset)}>
                  <Text as="span" flex="1" minW="0" truncate>
                    {preset.label}
                  </Text>
                  <IconButton
                    aria-label={t('widgets.generate.deletePresetNamed', { name: preset.label })}
                    color="fg.muted"
                    size="2xs"
                    variant="ghost"
                    onClick={(event) => removePreset(event, preset.id)}
                  >
                    <Trash2Icon />
                  </IconButton>
                </Menu.Item>
              ))}
            </MenuContent>
          </Menu.Positioner>
        </Portal>
      </Menu.Root>
      <Tooltip content={resetLabel}>
        <IconButton
          aria-label={resetLabel}
          color="fg.muted"
          disabled={!selectedModel || isAtModelDefaults}
          size="2xs"
          variant="ghost"
          onClick={resetToModelDefaults}
        >
          <Icon as={RotateCcwIcon} boxSize="3.5" />
        </IconButton>
      </Tooltip>
      <RenameDialog
        initialName=""
        isOpen={isSavingPreset}
        label={t('widgets.generate.presetName')}
        submitLabel={t('widgets.generate.savePresetAction')}
        title={t('widgets.generate.savePresetTitle')}
        onClose={() => setIsSavingPreset(false)}
        onSubmit={savePreset}
      />
    </>
  );
};
