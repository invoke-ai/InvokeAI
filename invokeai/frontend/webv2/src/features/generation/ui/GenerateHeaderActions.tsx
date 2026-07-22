/* oxlint-disable react-perf/jsx-no-new-function-as-prop */
import type { GenerationModelCatalogItem as ModelConfig } from '@features/generation/contracts';
import type { VaeModelConfig } from '@features/generation/core/types';

import { Icon } from '@chakra-ui/react';
import { isSupportedGenerateModel } from '@features/generation/core/baseGenerationPolicies';
import { normalizeGenerateSettings } from '@features/generation/core/settings';
import { IconButton, Tooltip } from '@platform/ui';
import { RotateCcwIcon } from 'lucide-react';
import { useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { flushGenerateDrafts } from './generateDraftRegistry';
import { useGenerationUi } from './GenerationUiContext';
import {
  getModelDefaultsPatch,
  getModelDefaultSettings,
  settingsMatchModelDefaults,
} from './shared/modelDefaultSettings';

/**
 * Widget-header action: reset every model-governed setting (dimensions, steps,
 * guidance, scheduler, VAE, LoRA enablement) back to the selected model's
 * defaults. Prompts and other non-default-bearing fields are left untouched.
 */
export const GenerateHeaderActions = () => {
  const { t } = useTranslation();
  const ui = useGenerationUi();
  const models = ui.models.catalog;
  const settings = normalizeGenerateSettings(ui.project.generateValues);
  const selectedModel = useMemo(
    () => models.filter(isSupportedGenerateModel).find((model) => model.key === settings?.modelKey),
    [models, settings?.modelKey]
  );
  const vaeModels = useMemo(
    () => models.filter((model): model is ModelConfig & VaeModelConfig => model.type === 'vae'),
    [models]
  );
  const modelDefaultSettings =
    settings && selectedModel ? getModelDefaultSettings(settings, selectedModel, vaeModels) : null;
  const isAtModelDefaults =
    settings && modelDefaultSettings ? settingsMatchModelDefaults(settings, modelDefaultSettings) : false;
  const label = t('widgets.generate.resetAllToModelDefaults');

  const resetToModelDefaults = () => {
    if (!selectedModel || !settings) {
      return;
    }

    // Flush pending debounced edits first so this patch lands on top of them;
    // the patch only carries model-governed keys, so flushed prompt edits survive.
    flushGenerateDrafts();
    ui.settings.patchGenerateSettings(
      getModelDefaultsPatch(settings, selectedModel, vaeModels),
      ui.project.activeProjectId
    );
  };

  return (
    <Tooltip content={label}>
      <IconButton
        aria-label={label}
        color="fg.muted"
        disabled={!selectedModel || isAtModelDefaults}
        size="2xs"
        variant="ghost"
        onClick={resetToModelDefaults}
      >
        <Icon as={RotateCcwIcon} boxSize="3.5" />
      </IconButton>
    </Tooltip>
  );
};
