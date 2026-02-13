import { IconButton } from '@invoke-ai/ui-library';
import { toast } from 'features/toast/toast';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiDownloadSimpleBold } from 'react-icons/pi';
import type { AnyModelConfig } from 'services/api/types';

type Props = {
  modelConfig: AnyModelConfig;
};

const buildExportData = (modelConfig: AnyModelConfig): Record<string, unknown> => {
  const data: Record<string, unknown> = {};

  if ('default_settings' in modelConfig && modelConfig.default_settings) {
    data.default_settings = modelConfig.default_settings;
  }

  if ('trigger_phrases' in modelConfig && modelConfig.trigger_phrases) {
    data.trigger_phrases = modelConfig.trigger_phrases;
  }

  if ('cpu_only' in modelConfig && modelConfig.cpu_only != null) {
    data.cpu_only = modelConfig.cpu_only;
  }

  return data;
};

const sanitizeFilename = (name: string): string => {
  return name.replace(/[<>:"/\\|?*]/g, '_');
};

export const ModelSettingsExportButton = memo(({ modelConfig }: Props) => {
  const { t } = useTranslation();

  const handleExport = useCallback(() => {
    const data = buildExportData(modelConfig);
    const json = JSON.stringify(data, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const filename = `${sanitizeFilename(modelConfig.name)}.json`;

    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    toast({
      id: 'SETTINGS_EXPORTED',
      title: t('modelManager.settingsExported'),
      status: 'success',
    });
  }, [modelConfig, t]);

  return (
    <IconButton
      size="sm"
      icon={<PiDownloadSimpleBold />}
      aria-label={t('modelManager.exportSettings')}
      tooltip={t('modelManager.exportSettings')}
      onClick={handleExport}
    />
  );
});

ModelSettingsExportButton.displayName = 'ModelSettingsExportButton';
