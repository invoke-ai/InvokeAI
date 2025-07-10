import { Button, Collapse, Flex, Icon, Text } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import { fixTooltipCloseOnScrollStyles } from 'common/util/fixTooltipCloseOnScrollStyles';
import {
  collapsedSectionToggled,
  selectCollapsedSections,
  selectStylePresetSearchTerm,
} from 'features/stylePresets/store/stylePresetSlice';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCaretDownBold } from 'react-icons/pi';
import type { StylePresetRecordWithImage } from 'services/api/endpoints/stylePresets';

import { StylePresetListItem } from './StylePresetListItem';

export const StylePresetList = ({ title, data }: { title: string; data: StylePresetRecordWithImage[] }) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const searchTerm = useAppSelector(selectStylePresetSearchTerm);
  const collapsedSections = useAppSelector(selectCollapsedSections);

  // Determine which section this is based on the title
  const getSectionKey = useCallback(
    (title: string) => {
      if (title === t('stylePresets.myTemplates')) {
        return 'myTemplates';
      }
      if (title === t('stylePresets.sharedTemplates')) {
        return 'sharedTemplates';
      }
      if (title === t('stylePresets.defaultTemplates')) {
        return 'defaultTemplates';
      }
      return 'myTemplates'; // fallback
    },
    [t]
  );

  const sectionKey = getSectionKey(title);
  const isOpen = !collapsedSections[sectionKey];

  const handleToggle = useCallback(() => {
    dispatch(collapsedSectionToggled(sectionKey));
  }, [dispatch, sectionKey]);

  return (
    <Flex flexDir="column">
      <Button variant="unstyled" onClick={handleToggle}>
        <Flex gap={2} alignItems="center">
          <Icon boxSize={4} as={PiCaretDownBold} transform={isOpen ? undefined : 'rotate(-90deg)'} fill="base.500" />
          <Text fontSize="sm" fontWeight="semibold" userSelect="none" color="base.500">
            {title}
          </Text>
        </Flex>
      </Button>
      <Collapse in={isOpen} style={fixTooltipCloseOnScrollStyles}>
        {data.length ? (
          data.map((preset) => <StylePresetListItem preset={preset} key={preset.id} />)
        ) : (
          <IAINoContentFallback
            fontSize="sm"
            py={4}
            label={searchTerm ? t('stylePresets.noMatchingTemplates') : t('stylePresets.noTemplates')}
            icon={null}
          />
        )}
      </Collapse>
    </Flex>
  );
};
