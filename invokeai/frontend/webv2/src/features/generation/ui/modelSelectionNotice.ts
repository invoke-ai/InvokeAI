import type { GenerateModelSelectionResult } from '@features/generation/core/baseGenerationPolicies';
import type { TFunction } from 'i18next';

interface GenerateModelSelectionNotifier {
  info(title: string, message?: string): void;
}

export const notifyGenerateModelSelectionCleared = ({
  clearedLabels,
  locale,
  modelName,
  notifications,
  reason = 'incompatible',
  t,
}: {
  clearedLabels: readonly string[];
  locale: string | undefined;
  modelName: string;
  notifications: GenerateModelSelectionNotifier;
  reason?: 'incompatible' | 'preset';
  t: TFunction;
}): void => {
  if (clearedLabels.length === 0) {
    return;
  }

  const labels = new Intl.ListFormat(locale, { style: 'long', type: 'conjunction' }).format(clearedLabels);

  notifications.info(
    t(reason === 'preset' ? 'widgets.generate.presetSettingsCleared' : 'widgets.generate.incompatibleSettingsCleared'),
    t(
      reason === 'preset'
        ? 'widgets.generate.presetSettingsClearedDescription'
        : 'widgets.generate.incompatibleSettingsClearedDescription',
      {
        count: clearedLabels.length,
        labels,
        name: modelName,
      }
    )
  );
};

/** A switch that only moves the size is not confirmed, so this names the size it moved to. */
export const notifyGenerateDimensionsAdjusted = ({
  modelName,
  notifications,
  result,
  t,
}: {
  modelName: string;
  notifications: GenerateModelSelectionNotifier;
  result: Pick<GenerateModelSelectionResult, 'dimensionsAdjusted' | 'settings'>;
  t: TFunction;
}): void => {
  if (!result.dimensionsAdjusted) {
    return;
  }

  notifications.info(
    t('widgets.generate.dimensionsAdjusted'),
    t('widgets.generate.dimensionsAdjustedDescription', {
      height: result.settings.height,
      name: modelName,
      width: result.settings.width,
    })
  );
};
