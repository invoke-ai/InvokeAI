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
