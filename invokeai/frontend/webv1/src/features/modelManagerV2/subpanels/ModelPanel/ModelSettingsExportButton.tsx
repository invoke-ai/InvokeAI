import { IconButton } from '@invoke-ai/ui-library';
import { toast } from 'features/toast/toast';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiDownloadSimpleBold } from 'react-icons/pi';
import type { AnyModelConfigWithExternal } from 'services/api/types';

import { buildExportData, fetchImageAsDataUrl, sanitizeFilename } from './modelSettingsIO';

type Props = {
  modelConfig: AnyModelConfigWithExternal;
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
