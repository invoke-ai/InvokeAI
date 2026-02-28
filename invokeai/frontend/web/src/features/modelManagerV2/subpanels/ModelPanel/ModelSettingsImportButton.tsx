import { IconButton } from '@invoke-ai/ui-library';
import { toast } from 'features/toast/toast';
import type { ChangeEvent } from 'react';
import { memo, useCallback, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { PiUploadSimpleBold } from 'react-icons/pi';
import { useUpdateModelMutation } from 'services/api/endpoints/models';
import type { AnyModelConfig } from 'services/api/types';

const validateImportData = (data: unknown): data is Record<string, unknown> => {
  if (typeof data !== 'object' || data === null || Array.isArray(data)) {
    return false;
  }

  const obj = data as Record<string, unknown>;

  if ('trigger_phrases' in obj && obj.trigger_phrases !== undefined) {
    if (!Array.isArray(obj.trigger_phrases) || !obj.trigger_phrases.every((p) => typeof p === 'string')) {
      return false;
    }
  }

  if ('default_settings' in obj && obj.default_settings !== undefined) {
    if (
      typeof obj.default_settings !== 'object' ||
      obj.default_settings === null ||
      Array.isArray(obj.default_settings)
    ) {
      return false;
    }
  }

  if ('cpu_only' in obj && obj.cpu_only !== undefined) {
    if (typeof obj.cpu_only !== 'boolean') {
      return false;
    }
  }

  return true;
};

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
      const skippedFields: string[] = [];

      const importableFields = ['default_settings', 'trigger_phrases', 'cpu_only'] as const;

      for (const field of importableFields) {
        if (!(field in data) || data[field] === undefined || data[field] === null) {
          continue;
        }
        if (field in modelConfig) {
          body[field] = data[field];
        } else {
          skippedFields.push(field);
        }
      }

      if (Object.keys(body).length === 0) {
        if (skippedFields.length > 0) {
          toast({
            id: 'SETTINGS_IMPORT_INCOMPATIBLE',
            title: t('modelManager.settingsImportIncompatible'),
            status: 'warning',
          });
        }
        return;
      }

      await updateModel({
        key: modelConfig.key,
        body,
      })
        .unwrap()
        .then(() => {
          if (skippedFields.length > 0) {
            toast({
              id: 'SETTINGS_IMPORTED',
              title: t('modelManager.settingsImportedPartial', { fields: skippedFields.join(', ') }),
              status: 'warning',
            });
          } else {
            toast({
              id: 'SETTINGS_IMPORTED',
              title: t('modelManager.settingsImported'),
              status: 'success',
            });
          }
        })
        .catch((_error) => {
          toast({
            id: 'SETTINGS_IMPORT_FAILED',
            title: t('modelManager.settingsImportFailed'),
            status: 'error',
          });
        });
    },
    [modelConfig, updateModel, t]
  );

  const handleFileChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (!file) {
        return;
      }

      const reader = new FileReader();
      reader.onload = (event) => {
        try {
          const json = JSON.parse(event.target?.result as string);
          if (!validateImportData(json)) {
            toast({
              id: 'SETTINGS_IMPORT_INVALID',
              title: t('modelManager.settingsImportInvalidFile'),
              status: 'error',
            });
            return;
          }
          applySettings(json);
        } catch {
          toast({
            id: 'SETTINGS_IMPORT_INVALID',
            title: t('modelManager.settingsImportInvalidFile'),
            status: 'error',
          });
        }
      };
      reader.onerror = () => {
        toast({
          id: 'SETTINGS_IMPORT_INVALID',
          title: t('modelManager.settingsImportInvalidFile'),
          status: 'error',
        });
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
      <input ref={fileInputRef} type="file" accept=".json" onChange={handleFileChange} style={{ display: 'none' }} />
    </>
  );
});

ModelSettingsImportButton.displayName = 'ModelSettingsImportButton';
