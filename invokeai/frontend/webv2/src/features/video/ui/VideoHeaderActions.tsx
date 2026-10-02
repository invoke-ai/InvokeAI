/* oxlint-disable react-perf/jsx-no-new-function-as-prop */
import { Icon } from '@chakra-ui/react';
import { useModelsSelector } from '@features/models';
import { normalizeVideoWidgetValues } from '@features/video/core/settings';
import { getVideoSettingsWithModelDefaults } from '@features/video/core/videoPolicies';
import { syncVideoWidgetValuesWithModels } from '@features/video/core/widgetValues';
import { IconButton, Tooltip } from '@platform/ui';
import { RotateCcwIcon } from 'lucide-react';
import { useTranslation } from 'react-i18next';

import { areVideoValuesEqual } from './videoComparators';
import { useVideoUi } from './VideoUiContext';

const sortLorasByKey = <T extends { model: { key: string } }>(loras: readonly T[]): T[] =>
  [...loras].sort((a, b) => a.model.key.localeCompare(b.model.key));

export const VideoHeaderActions = () => {
  const { t } = useTranslation();
  const { patchValues, rawValues } = useVideoUi();
  const models = useModelsSelector((snapshot) => snapshot.models);
  // Wait for authoritative catalog data so reset cannot remove a valid accelerator pair.
  const modelsLoaded = useModelsSelector((snapshot) => snapshot.status) === 'loaded';
  // Match the panel's reconciled model; raw state may still reference an uninstalled model.
  const normalized = normalizeVideoWidgetValues(rawValues);
  const values = normalized && modelsLoaded ? syncVideoWidgetValuesWithModels(normalized, models) : normalized;
  const model = values?.model ?? null;
  const defaults =
    modelsLoaded && values && model ? { ...getVideoSettingsWithModelDefaults(values, model, models), model } : null;
  // LoRA order is insertion history, not a model-governed setting; compare as sets.
  const isAtModelDefaults =
    values && defaults
      ? areVideoValuesEqual(
          { ...values, loras: sortLorasByKey(values.loras) },
          { ...defaults, loras: sortLorasByKey(defaults.loras) }
        )
      : true;
  const label = t('widgets.video.resetAllToModelDefaults');

  const resetToModelDefaults = () => {
    if (defaults) {
      patchValues({ ...defaults });
    }
  };

  return (
    <Tooltip content={label}>
      <IconButton
        aria-label={label}
        disabled={isAtModelDefaults}
        size="2xs"
        variant="ghost"
        onClick={resetToModelDefaults}
      >
        <Icon as={RotateCcwIcon} boxSize="3.5" />
      </IconButton>
    </Tooltip>
  );
};
