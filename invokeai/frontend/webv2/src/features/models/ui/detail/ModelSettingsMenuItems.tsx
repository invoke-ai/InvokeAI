/* eslint-disable react-perf/jsx-no-new-function-as-prop */
import { Icon, Menu } from '@chakra-ui/react';
import {
  buildExportData,
  dataUrlToFile,
  isImageDataUrl,
  partitionImportableFields,
  sanitizeFilename,
  validateImportData,
} from '@features/models/core/modelSettingsIO';
import { getModelImageUrl, updateModel, updateModelImage } from '@features/models/data/api';
import { getModelsSnapshot, markCoverImageChanged, replaceModelInStore } from '@features/models/data/modelsStore';
import { useNotify } from '@features/models/ui/useModelsNotify';
import { useScopedAction } from '@platform/react/useScopedAction';
import { assertAccountScopeCurrent } from '@platform/state/accountLifecycle';
import { DownloadIcon, UploadIcon } from 'lucide-react';
import { useRef, type ChangeEvent } from 'react';
import { useTranslation } from 'react-i18next';

/**
 * Export/import a model's user-editable settings as JSON (format shared with
 * the legacy frontend). Rendered inside the detail pane's overflow menu; the
 * full config is read from the store at action time so the menu itself only
 * needs a key.
 */
export const ModelSettingsMenuItems = ({ modelKey }: { modelKey: string }) => {
  const { t } = useTranslation();
  const notify = useNotify();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { run } = useScopedAction();

  const findModel = () => getModelsSnapshot().models.find((candidate) => candidate.key === modelKey);

  const handleExport = async () => {
    const model = findModel();

    if (!model) {
      return;
    }

    const data = buildExportData(model);

    if (model.cover_image) {
      const dataUrl = await fetchImageAsDataUrl(getModelImageUrl(model.key));

      if (dataUrl) {
        data.cover_image = dataUrl;
      }
    }

    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');

    anchor.href = url;
    anchor.download = `${sanitizeFilename(model.name)}.json`;
    anchor.click();
    URL.revokeObjectURL(url);
    notify.success(t('models.settingsExported'), model.name);
  };

  const applyImport = (data: Record<string, unknown>) =>
    run(
      async (owner) => {
        const model = findModel();

        if (!model) {
          return;
        }

        const { body, skipped } = partitionImportableFields(data, model);
        const coverImageDataUrl =
          typeof data.cover_image === 'string' && isImageDataUrl(data.cover_image) ? data.cover_image : null;

        if (Object.keys(body).length === 0 && !coverImageDataUrl) {
          notify.error(t('models.settingsImportIncompatible'), skipped.join(', '));

          return;
        }

        let applied = false;

        if (Object.keys(body).length > 0) {
          const updated = await updateModel(model.key, body, owner.signal);

          assertAccountScopeCurrent(owner);
          replaceModelInStore(updated);
          applied = true;
        }

        if (coverImageDataUrl) {
          const imageFile = dataUrlToFile(coverImageDataUrl, `${model.key}.png`);

          if (imageFile) {
            try {
              await updateModelImage(model.key, imageFile, owner.signal);
              assertAccountScopeCurrent(owner);
              markCoverImageChanged(model.key, true);
              applied = true;
            } catch {
              assertAccountScopeCurrent(owner);
              skipped.push('cover_image');
            }
          } else {
            skipped.push('cover_image');
          }
        }

        if (!applied) {
          throw new Error(t('models.settingsImportFailed'));
        }

        if (skipped.length > 0) {
          notify.info(t('models.settingsImportedPartial', { fields: skipped.join(', ') }));
        } else {
          notify.success(t('models.settingsImported'), model.name);
        }
      },
      (message) => notify.error(t('models.settingsImportFailed'), message)
    );

  const handleFileChange = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.currentTarget.files?.[0];

    // Reset so re-picking the same file fires change again.
    event.currentTarget.value = '';

    if (!file) {
      return;
    }

    let parsed: unknown;

    try {
      parsed = JSON.parse(await file.text());
    } catch {
      notify.error(t('models.settingsImportInvalidFile'));

      return;
    }

    if (!validateImportData(parsed)) {
      notify.error(t('models.settingsImportInvalidFile'));

      return;
    }

    await applyImport(parsed);
  };

  return (
    <>
      <Menu.Item value="export-settings" onClick={() => void handleExport()}>
        <Icon as={DownloadIcon} boxSize="3.5" />
        <Menu.ItemText fontSize="xs">{t('models.exportSettings')}</Menu.ItemText>
      </Menu.Item>
      <Menu.Item value="import-settings" onClick={() => fileInputRef.current?.click()}>
        <Icon as={UploadIcon} boxSize="3.5" />
        <Menu.ItemText fontSize="xs">{t('models.importSettings')}</Menu.ItemText>
      </Menu.Item>
      <input
        accept="application/json,.json"
        hidden
        ref={fileInputRef}
        type="file"
        onChange={(event) => void handleFileChange(event)}
      />
    </>
  );
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
