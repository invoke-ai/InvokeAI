import { IconButton } from '@invoke-ai/ui-library';
import { toast } from 'features/toast/toast';
import type { ChangeEvent } from 'react';
import { memo, useCallback, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { PiUploadSimpleBold } from 'react-icons/pi';
import { useUpdateModelImageMutation, useUpdateModelMutation } from 'services/api/endpoints/models';
import type { AnyModelConfigWithExternal } from 'services/api/types';

const isSafeUrl = (url: string): boolean => {
  return url.startsWith('https://') || url.startsWith('http://');
};

const isImageDataUrl = (value: string): boolean => {
  return /^data:image\/[a-zA-Z0-9.+-]+;base64,/.test(value);
};

const dataUrlToFile = (dataUrl: string, filename: string): File | null => {
  const match = dataUrl.match(/^data:([^;]+);base64,(.+)$/);
  if (!match) {
    return null;
  }
  const mime = match[1];
  const b64 = match[2];
  if (!mime || !b64) {
    return null;
  }
  try {
    const binary = atob(b64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) {
      bytes[i] = binary.charCodeAt(i);
    }
    return new File([bytes], filename, { type: mime });
  } catch {
    return null;
  }
};

const validateImportData = (data: unknown): data is Record<string, unknown> => {
  if (typeof data !== 'object' || data === null || Array.isArray(data)) {
    return false;
  }

  const obj = data as Record<string, unknown>;

  if ('name' in obj && obj.name !== undefined && obj.name !== null) {
    if (typeof obj.name !== 'string') {
      return false;
    }
  }

  if ('description' in obj && obj.description !== undefined && obj.description !== null) {
    if (typeof obj.description !== 'string') {
      return false;
    }
  }

  if ('source_url' in obj && obj.source_url !== undefined && obj.source_url !== null) {
    if (typeof obj.source_url !== 'string') {
      return false;
    }
    if (obj.source_url.length > 0 && !isSafeUrl(obj.source_url)) {
      return false;
    }
  }

  if ('cover_image' in obj && obj.cover_image !== undefined && obj.cover_image !== null) {
    if (typeof obj.cover_image !== 'string' || !isImageDataUrl(obj.cover_image)) {
      return false;
    }
  }

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
  modelConfig: AnyModelConfigWithExternal;
};

export const ModelSettingsImportButton = memo(({ modelConfig }: Props) => {
  const { t } = useTranslation();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [updateModel] = useUpdateModelMutation();
  const [updateModelImage] = useUpdateModelImageMutation();

  const applySettings = useCallback(
    async (data: Record<string, unknown>) => {
      const body: Record<string, unknown> = {};
      const skippedFields: string[] = [];

      const importableFields = [
        'name',
        'description',
        'source_url',
        'default_settings',
        'trigger_phrases',
        'cpu_only',
      ] as const;

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

      const coverImageDataUrl =
        'cover_image' in data && typeof data.cover_image === 'string' && isImageDataUrl(data.cover_image)
          ? data.cover_image
          : null;

      if (Object.keys(body).length === 0 && !coverImageDataUrl) {
        if (skippedFields.length > 0) {
          toast({
            id: 'SETTINGS_IMPORT_INCOMPATIBLE',
            title: t('modelManager.settingsImportIncompatible'),
            status: 'warning',
          });
        }
        return;
      }

      let appliedAnything = false;
      if (Object.keys(body).length > 0) {
        try {
          await updateModel({
            key: modelConfig.key,
            body,
          }).unwrap();
          appliedAnything = true;
        } catch {
          toast({
            id: 'SETTINGS_IMPORT_FAILED',
            title: t('modelManager.settingsImportFailed'),
            status: 'error',
          });
          return;
        }
      }

      if (coverImageDataUrl) {
        const imageFile = dataUrlToFile(coverImageDataUrl, `${modelConfig.key}.png`);
        if (!imageFile) {
          skippedFields.push('cover_image');
        } else {
          try {
            await updateModelImage({ key: modelConfig.key, image: imageFile }).unwrap();
            appliedAnything = true;
          } catch {
            skippedFields.push('cover_image');
          }
        }
      }

      if (!appliedAnything) {
        toast({
          id: 'SETTINGS_IMPORT_FAILED',
          title: t('modelManager.settingsImportFailed'),
          status: 'error',
        });
        return;
      }

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
    },
    [modelConfig, updateModel, updateModelImage, t]
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
