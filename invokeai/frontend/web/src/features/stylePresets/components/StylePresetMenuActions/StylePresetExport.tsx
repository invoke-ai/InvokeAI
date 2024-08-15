import { MenuItem } from '@invoke-ai/ui-library';
import { toast } from 'features/toast/toast';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiDownloadSimpleBold, PiSpinner } from 'react-icons/pi';
import { useLazyExportStylePresetsQuery } from 'services/api/endpoints/stylePresets';

export const StylePresetExport = ({ onClose }: { onClose: () => void }) => {
  const [exportStylePresets, { isLoading }] = useLazyExportStylePresetsQuery();
  const { t } = useTranslation();

  const handleClickDownloadCsv = useCallback(async () => {
    let blob;
    try {
      const response = await exportStylePresets().unwrap();
      blob = new Blob([response], { type: 'text/csv' });
    } catch (error) {
      toast({
        status: 'error',
        title: t('stylePresets.exportFailed'),
      });
    } finally {
      onClose();
    }

    if (blob) {
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'data.csv';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);
      toast({
        status: 'success',
        title: t('stylePresets.exportDownloaded'),
      });
    }
  }, [exportStylePresets, onClose, t]);

  return (
    <MenuItem icon={!isLoading ? <PiDownloadSimpleBold /> : <PiSpinner />} onClickCapture={handleClickDownloadCsv}>
      {t('stylePresets.downloadCsv')}
    </MenuItem>
  );
};
