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
import { GeneratePresetsPopover } from './GeneratePresetsPopover';
import { useGenerateValues, useGenerationUi } from './GenerationUiContext';
import {
  getModelDefaultsPatch,
  getModelDefaultSettings,
  settingsMatchModelDefaults,
} from './shared/modelDefaultSettings';

export const GenerateHeaderActions = () => {
  const { t } = useTranslation();
  const ui = useGenerationUi();
  const models = ui.models.catalog;
  const projectId = ui.project.activeProjectId;
  const supportedModels = useMemo(() => models.filter(isSupportedGenerateModel), [models]);
  const vaeModels = useMemo(
    () => models.filter((model): model is ModelConfig & VaeModelConfig => model.type === 'vae'),
    [models]
  );
  const resolveDefaults = (values: Record<string, unknown>) => {
    const settings = normalizeGenerateSettings(values);
    const selectedModel = settings && supportedModels.find((model) => model.key === settings.modelKey);

    return settings && selectedModel ? { selectedModel, settings } : null;
  };
  // Select only what the button shows, so ordinary edits do not re-render the header.
  const { hasSelectedModel, isAtModelDefaults } = useGenerateValues((values) => {
    const resolved = resolveDefaults(values);

    return {
      hasSelectedModel: resolved !== null,
      isAtModelDefaults:
        resolved !== null &&
        settingsMatchModelDefaults(
          resolved.settings,
          getModelDefaultSettings(resolved.settings, resolved.selectedModel, vaeModels)
        ),
    };
  });

  const resetToModelDefaults = () => {
    // Flush before reading so reset preserves pending prompt edits.
    flushGenerateDrafts();
    const resolved = resolveDefaults(ui.generateValues.getSnapshot());

    if (resolved) {
      ui.settings.patchGenerateSettings(
        getModelDefaultsPatch(resolved.settings, resolved.selectedModel, vaeModels),
        projectId
      );
    }
  };

  const resetLabel = t('widgets.generate.resetAllToModelDefaults');

  return (
    <>
      <GeneratePresetsPopover />
      <Tooltip content={resetLabel}>
        <IconButton
          aria-label={resetLabel}
          color="fg.muted"
          disabled={!hasSelectedModel || isAtModelDefaults}
          size="2xs"
          variant="ghost"
          onClick={resetToModelDefaults}
        >
          <Icon as={RotateCcwIcon} boxSize="3.5" />
        </IconButton>
      </Tooltip>
    </>
  );
};
