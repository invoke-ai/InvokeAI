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

/**
 * Widget-header action: reset every default-bearing setting (frames, fps,
 * target resolution, steps, CFG, accelerator state, aspect ratio, component
 * selections) back to the selected model's defaults. Prompts and conditioning
 * media are left untouched.
 */
const sortLorasByKey = <T extends { model: { key: string } }>(loras: readonly T[]): T[] =>
  [...loras].sort((a, b) => a.model.key.localeCompare(b.model.key));

export const VideoHeaderActions = () => {
  const { t } = useTranslation();
  const { patchValues, rawValues } = useVideoUi();
  const models = useModelsSelector((snapshot) => snapshot.models);
  // Defaults computed from an unloaded catalog would claim the accelerator
  // pair is "not a default" and strip it as the reset — stay disabled until
  // the catalog is authoritative.
  const modelsLoaded = useModelsSelector((snapshot) => snapshot.status) === 'loaded';
  // Sync against the catalog exactly like the panel view does: computing from
  // the raw store would judge (and reset to!) a phantom model the view no
  // longer shows — e.g. one uninstalled while the panel was open.
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
