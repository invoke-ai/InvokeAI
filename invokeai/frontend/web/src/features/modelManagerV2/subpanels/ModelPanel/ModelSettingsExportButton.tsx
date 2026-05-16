import { IconButton } from '@invoke-ai/ui-library';
import { toast } from 'features/toast/toast';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiDownloadSimpleBold } from 'react-icons/pi';
import type { AnyModelConfigWithExternal } from 'services/api/types';

type Props = {
  modelConfig: AnyModelConfigWithExternal;
};

const buildExportData = (modelConfig: AnyModelConfigWithExternal): Record<string, unknown> => {
  const data: Record<string, unknown> = {};

  if ('name' in modelConfig && typeof modelConfig.name === 'string' && modelConfig.name.length > 0) {
    data.name = modelConfig.name;
  }

  if (
    'description' in modelConfig &&
    typeof modelConfig.description === 'string' &&
    modelConfig.description.length > 0
  ) {
    data.description = modelConfig.description;
  }

  if ('source_url' in modelConfig && typeof modelConfig.source_url === 'string' && modelConfig.source_url.length > 0) {
    data.source_url = modelConfig.source_url;
  }

  if (
    'default_settings' in modelConfig &&
    modelConfig.default_settings !== undefined &&
    modelConfig.default_settings !== null
  ) {
    data.default_settings = modelConfig.default_settings;
  }

  if (
    'trigger_phrases' in modelConfig &&
    modelConfig.trigger_phrases !== undefined &&
    modelConfig.trigger_phrases !== null
  ) {
    data.trigger_phrases = modelConfig.trigger_phrases;
  }

  if ('cpu_only' in modelConfig && modelConfig.cpu_only !== null) {
    data.cpu_only = modelConfig.cpu_only;
  }

  return data;
};

const sanitizeFilename = (name: string): string => {
  return name.replace(/[<>:"/\\|?*]/g, '_');
};

const fetchImageAsDataUrl = async (url: string): Promise<string | null> => {
  try {
    const response = await fetch(url);
    if (!response.ok) {
      return null;
    }
    const blob = await response.blob();
    if (!blob.type.startsWith('image/')) {
      return null;
    }
    return await new Promise<string | null>((resolve) => {
      const reader = new FileReader();
      reader.onload = () => resolve(typeof reader.result === 'string' ? reader.result : null);
      reader.onerror = () => resolve(null);
      reader.readAsDataURL(blob);
    });
  } catch {
    return null;
  }
};

export const ModelSettingsExportButton = memo(({ modelConfig }: Props) => {
  const { t } = useTranslation();

  const handleExport = useCallback(async () => {
    const data = buildExportData(modelConfig);

    if (
      'cover_image' in modelConfig &&
      typeof modelConfig.cover_image === 'string' &&
      modelConfig.cover_image.length > 0
    ) {
      const dataUrl = await fetchImageAsDataUrl(modelConfig.cover_image);
      if (dataUrl) {
        data.cover_image = dataUrl;
      }
    }

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
