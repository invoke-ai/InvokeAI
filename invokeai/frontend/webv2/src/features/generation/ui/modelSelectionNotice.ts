import type { TFunction } from 'i18next';

interface GenerateModelSelectionNotifier {
  info(title: string, message?: string): void;
}

export const notifyGenerateModelSelectionCleared = ({
  clearedLabels,
  locale,
  modelName,
  notifications,
  t,
}: {
  clearedLabels: readonly string[];
  locale: string | undefined;
  modelName: string;
  notifications: GenerateModelSelectionNotifier;
  t: TFunction;
}): void => {
  if (clearedLabels.length === 0) {
    return;
  }

  const labels = new Intl.ListFormat(locale, { style: 'long', type: 'conjunction' }).format(clearedLabels);

  notifications.info(
    t('widgets.generate.incompatibleSettingsCleared'),
    t('widgets.generate.incompatibleSettingsClearedDescription', {
      count: clearedLabels.length,
      labels,
      name: modelName,
    })
  );
};
