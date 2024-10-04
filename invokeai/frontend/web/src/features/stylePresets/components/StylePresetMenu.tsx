import { Flex } from '@invoke-ai/ui-library';
import { EMPTY_ARRAY } from 'app/store/constants';
import { useAppSelector } from 'app/store/storeHooks';
import { StylePresetExportButton } from 'features/stylePresets/components/StylePresetExportButton';
import { StylePresetImportButton } from 'features/stylePresets/components/StylePresetImportButton';
import { selectStylePresetSearchTerm } from 'features/stylePresets/store/stylePresetSlice';
import { selectAllowPrivateStylePresets } from 'features/system/store/configSlice';
import { useTranslation } from 'react-i18next';
import type { StylePresetRecordWithImage } from 'services/api/endpoints/stylePresets';
import { useListStylePresetsQuery } from 'services/api/endpoints/stylePresets';

import { StylePresetCreateButton } from './StylePresetCreateButton';
import { StylePresetList } from './StylePresetList';
import StylePresetSearch from './StylePresetSearch';

export const StylePresetMenu = () => {
  const searchTerm = useAppSelector(selectStylePresetSearchTerm);
  const allowPrivateStylePresets = useAppSelector(selectAllowPrivateStylePresets);
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

  return (
    <Flex flexDir="column" gap={2} padding={3} layerStyle="second" borderRadius="base">
      <Flex alignItems="center" gap={2} w="full" justifyContent="space-between">
        <StylePresetSearch />
        <StylePresetCreateButton />
        <StylePresetImportButton />
        <StylePresetExportButton />
      </Flex>

      <StylePresetList title={t('stylePresets.myTemplates')} data={data.presets} />
      {allowPrivateStylePresets && (
        <StylePresetList title={t('stylePresets.sharedTemplates')} data={data.sharedPresets} />
      )}
      <StylePresetList title={t('stylePresets.defaultTemplates')} data={data.defaultPresets} />
    </Flex>
  );
};
