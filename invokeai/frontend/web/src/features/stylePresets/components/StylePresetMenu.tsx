import { Flex, IconButton, Text } from '@invoke-ai/ui-library';
import { EMPTY_ARRAY } from 'app/store/constants';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  isModalOpenChanged,
  prefilledFormDataChanged,
  updatingStylePresetIdChanged,
} from 'features/stylePresets/store/stylePresetModalSlice';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiPlusBold } from 'react-icons/pi';
import type { StylePresetRecordWithImage } from 'services/api/endpoints/stylePresets';
import { useListStylePresetsQuery } from 'services/api/endpoints/stylePresets';

import { StylePresetList } from './StylePresetList';
import StylePresetSearch from './StylePresetSearch';

export const StylePresetMenu = () => {
  const searchTerm = useAppSelector((s) => s.stylePreset.searchTerm);
  const { data } = useListStylePresetsQuery(undefined, {
    selectFromResult: ({ data, error, isLoading }) => {
      const filteredData =
        data?.filter((preset) => preset.name.toLowerCase().includes(searchTerm.toLowerCase())) || EMPTY_ARRAY;

      const groupedData = filteredData.reduce(
        (acc: { defaultPresets: StylePresetRecordWithImage[]; presets: StylePresetRecordWithImage[] }, preset) => {
          if (preset.is_default) {
            acc.defaultPresets.push(preset);
          } else {
            acc.presets.push(preset);
          }
          return acc;
        },
        { defaultPresets: [], presets: [] }
      );

      return {
        data: groupedData,
        error,
        isLoading,
      };
    },
  });

  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const handleClickAddNew = useCallback(() => {
    dispatch(prefilledFormDataChanged(null));
    dispatch(updatingStylePresetIdChanged(null));
    dispatch(isModalOpenChanged(true));
  }, [dispatch]);

  return (
    <Flex flexDir="column" gap="2" padding="10px" layerStyle="second">
      <Flex alignItems="center" gap="10" w="full" justifyContent="space-between">
        <StylePresetSearch />
        <IconButton
          icon={<PiPlusBold />}
          tooltip="Create Preset"
          aria-label="Create Preset"
          onClick={handleClickAddNew}
          size="md"
          variant="link"
          w={8}
          h={8}
        />
      </Flex>

      {data.presets.length === 0 && data.defaultPresets.length === 0 && (
        <Text m="20px" textAlign="center">
          {t('stylePresets.noMatchingTemplates')}
        </Text>
      )}

      <StylePresetList title={t('stylePresets.myTemplates')} data={data.presets} />

      <StylePresetList title={t('stylePresets.defaultTemplates')} data={data.defaultPresets} />
    </Flex>
  );
};
