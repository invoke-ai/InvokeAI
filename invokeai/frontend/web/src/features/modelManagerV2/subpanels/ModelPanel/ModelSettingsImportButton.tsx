import { IconButton } from '@invoke-ai/ui-library';
import { toast } from 'features/toast/toast';
import { memo, useCallback, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { PiUploadSimpleBold } from 'react-icons/pi';
import { useUpdateModelMutation } from 'services/api/endpoints/models';
import type { AnyModelConfig } from 'services/api/types';

type Props = {
  modelConfig: AnyModelConfig;
};

export const ModelSettingsImportButton = memo(({ modelConfig }: Props) => {
  const { t } = useTranslation();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [updateModel] = useUpdateModelMutation();

  const applySettings = useCallback(
    async (data: Record<string, unknown>) => {
      const body: Record<string, unknown> = {};

      if ('default_settings' in data && data.default_settings != null) {
        body.default_settings = data.default_settings;
      }

      if ('trigger_phrases' in data && Array.isArray(data.trigger_phrases)) {
        body.trigger_phrases = data.trigger_phrases;
      }

      if ('cpu_only' in data && data.cpu_only != null) {
        body.cpu_only = data.cpu_only;
      }

      if (Object.keys(body).length === 0) {
        return;
      }

      await updateModel({
        key: modelConfig.key,
        body,
      })
        .unwrap()
        .then(() => {
          toast({
            id: 'SETTINGS_IMPORTED',
            title: t('modelManager.settingsImported'),
            status: 'success',
          });
        })
        .catch((error) => {
          toast({
            id: 'SETTINGS_IMPORT_FAILED',
            title: `${t('modelManager.settingsImportFailed')}: ${error.data?.detail ?? ''}`,
            status: 'error',
          });
        });
    },
    [modelConfig.key, updateModel, t]
  );

  const handleFileChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (!file) {
        return;
      }

      const reader = new FileReader();
      reader.onload = (event) => {
        try {
          const json = JSON.parse(event.target?.result as string);
          applySettings(json);
        } catch {
          toast({
            id: 'SETTINGS_IMPORT_INVALID',
            title: t('modelManager.settingsImportInvalidFile'),
            status: 'error',
          });
        }
      };
      reader.readAsText(file);

      // Reset the input so the same file can be re-selected
      e.target.value = '';
    },
    [applySettings, t]
  );

  const handleClick = useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  return (
    <>
      <IconButton
        size="sm"
        icon={<PiUploadSimpleBold />}
        aria-label={t('modelManager.importSettings')}
        tooltip={t('modelManager.importSettings')}
        onClick={handleClick}
      />
      <input
        ref={fileInputRef}
        type="file"
        accept=".json"
        onChange={handleFileChange}
        style={{ display: 'none' }}
      />
    </>
  );
});

ModelSettingsImportButton.displayName = 'ModelSettingsImportButton';
