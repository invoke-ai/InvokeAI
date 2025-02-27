import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { IconButton, spinAnimation } from '@invoke-ai/ui-library';
import { EMPTY_ARRAY } from 'app/store/constants';
import { toast } from 'features/toast/toast';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiDownloadSimpleBold, PiSpinner } from 'react-icons/pi';
import { useLazyExportStylePresetsQuery, useListStylePresetsQuery } from 'services/api/endpoints/stylePresets';

const loadingStyles: SystemStyleObject = {
  svg: { animation: spinAnimation },
};

export const StylePresetExportButton = () => {
  const [exportStylePresets, { isLoading }] = useLazyExportStylePresetsQuery();
  const { t } = useTranslation();
  const { presetCount } = useListStylePresetsQuery(undefined, {
    selectFromResult: ({ data }) => {
      const presetsToExport = data?.filter((preset) => preset.type !== 'default') ?? EMPTY_ARRAY;
      return {
        presetCount: presetsToExport.length,
      };
    },
  });
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
      return;
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
  }, [exportStylePresets, t]);

  return (
    <IconButton
      onClick={handleClickDownloadCsv}
      icon={!isLoading ? <PiDownloadSimpleBold /> : <PiSpinner />}
      tooltip={t('stylePresets.exportPromptTemplates')}
      aria-label={t('stylePresets.exportPromptTemplates')}
      size="md"
      variant="link"
      w={8}
      h={8}
      sx={isLoading ? loadingStyles : undefined}
      isDisabled={isLoading || presetCount === 0}
    />
  );
};
