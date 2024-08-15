import { Flex, IconButton, Text } from '@invoke-ai/ui-library';
import { EMPTY_ARRAY } from 'app/store/constants';
import { useAppSelector } from 'app/store/storeHooks';
import { $stylePresetModalState } from 'features/stylePresets/store/stylePresetModal';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiDownloadBold, PiPlusBold } from 'react-icons/pi';
import type { StylePresetRecordWithImage } from 'services/api/endpoints/stylePresets';
import { useLazyExportStylePresetsQuery, useListStylePresetsQuery } from 'services/api/endpoints/stylePresets';

import { StylePresetList } from './StylePresetList';
import StylePresetSearch from './StylePresetSearch';
import { toast } from '../../toast/toast';

const generateCSV = (data: any[]) => {
  const header = ['Column1', 'Column2', 'Column3'];
  const csvRows = [
    header.join(','), // add header row
    ...data.map((row) => row.join(',')), // add data rows
  ];

  return csvRows.join('\n');
};

export const StylePresetMenu = () => {
  const searchTerm = useAppSelector((s) => s.stylePreset.searchTerm);
  const allowPrivateStylePresets = useAppSelector((s) => s.config.allowPrivateStylePresets);
  const { data } = useListStylePresetsQuery(undefined, {
    selectFromResult: ({ data }) => {
      const filteredData =
        data?.filter((preset) => preset.name.toLowerCase().includes(searchTerm.toLowerCase())) || EMPTY_ARRAY;

      const groupedData = filteredData.reduce(
        (
          acc: {
            defaultPresets: StylePresetRecordWithImage[];
            sharedPresets: StylePresetRecordWithImage[];
            presets: StylePresetRecordWithImage[];
          },
          preset
        ) => {
          if (preset.type === 'default') {
            acc.defaultPresets.push(preset);
          } else if (preset.type === 'project') {
            acc.sharedPresets.push(preset);
          } else {
            acc.presets.push(preset);
          }
          return acc;
        },
        { defaultPresets: [], sharedPresets: [], presets: [] }
      );

      return {
        data: groupedData,
      };
    },
  });

  const { t } = useTranslation();
  const [exportStylePresets, { isLoading }] = useLazyExportStylePresetsQuery();

  const handleClickAddNew = useCallback(() => {
    $stylePresetModalState.set({
      prefilledFormData: null,
      updatingStylePresetId: null,
      isModalOpen: true,
    });
  }, []);

  const handleClickDownloadCsv = useCallback(async () => {
    let blob;
    try {
      const response = await exportStylePresets().unwrap();
      blob = new Blob([response], { type: 'text/csv' });
    } catch (error) {
      toast({
        status: 'error',
        title: 'Unable to generate and download export',
      });
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
    }
    toast({
      status: 'success',
      title: 'Export downloaded',
    });
  }, [exportStylePresets]);

  return (
    <Flex flexDir="column" gap={2} padding={3} layerStyle="second" borderRadius="base">
      <Flex alignItems="center" gap={2} w="full" justifyContent="space-between">
        <StylePresetSearch />
        <IconButton
          icon={<PiDownloadBold />}
          tooltip={t('stylePresets.createPromptTemplate')}
          aria-label={t('stylePresets.createPromptTemplate')}
          onClick={handleClickDownloadCsv}
          size="md"
          variant="link"
          w={8}
          h={8}
          isDisabled={isLoading}
        />
        <IconButton
          icon={<PiPlusBold />}
          tooltip={t('stylePresets.createPromptTemplate')}
          aria-label={t('stylePresets.createPromptTemplate')}
          onClick={handleClickAddNew}
          size="md"
          variant="link"
          w={8}
          h={8}
        />
      </Flex>

      {data.presets.length === 0 && data.defaultPresets.length === 0 && (
        <Text p={10} textAlign="center">
          {t('stylePresets.noMatchingTemplates')}
        </Text>
      )}

      <StylePresetList title={t('stylePresets.myTemplates')} data={data.presets} />

      {allowPrivateStylePresets && (
        <StylePresetList title={t('stylePresets.sharedTemplates')} data={data.sharedPresets} />
      )}

      <StylePresetList title={t('stylePresets.defaultTemplates')} data={data.defaultPresets} />
    </Flex>
  );
};
