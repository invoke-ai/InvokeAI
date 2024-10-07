import { Button, Collapse, Flex, Icon, Text, useDisclosure } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import { selectStylePresetSearchTerm } from 'features/stylePresets/store/stylePresetSlice';
import { useTranslation } from 'react-i18next';
import { PiCaretDownBold } from 'react-icons/pi';
import type { StylePresetRecordWithImage } from 'services/api/endpoints/stylePresets';

import { StylePresetListItem } from './StylePresetListItem';

export const StylePresetList = ({ title, data }: { title: string; data: StylePresetRecordWithImage[] }) => {
  const { t } = useTranslation();
  const { onToggle, isOpen } = useDisclosure({ defaultIsOpen: true });
  const searchTerm = useAppSelector(selectStylePresetSearchTerm);

  return (
    <Flex flexDir="column">
      <Button variant="unstyled" onClick={onToggle}>
        <Flex gap={2} alignItems="center">
          <Icon boxSize={4} as={PiCaretDownBold} transform={isOpen ? undefined : 'rotate(-90deg)'} fill="base.500" />
          <Text fontSize="sm" fontWeight="semibold" userSelect="none" color="base.500">
            {title}
          </Text>
        </Flex>
      </Button>
      <Collapse in={isOpen}>
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
