import { Flex, IconButton, Menu, MenuButton, MenuList, useDisclosure } from '@invoke-ai/ui-library';
import { $stylePresetModalState } from 'features/stylePresets/store/stylePresetModal';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiDotsThreeOutlineFill, PiPlusBold } from 'react-icons/pi';

import { StylePresetExport } from './StylePresetExport';
import { StylePresetImport } from './StylePresetImport';

export const StylePresetMenuActions = () => {
  const { isOpen, onClose, onToggle } = useDisclosure();
  const handleClickAddNew = useCallback(() => {
    $stylePresetModalState.set({
      prefilledFormData: null,
      updatingStylePresetId: null,
      isModalOpen: true,
    });
  }, []);

  const { t } = useTranslation();

  return (
    <Flex alignItems="center" justifyContent="space-between">
      <IconButton
        icon={<PiPlusBold />}
        tooltip={t('stylePresets.createPromptTemplate')}
        aria-label={t('stylePresets.createPromptTemplate')}
        onClick={handleClickAddNew}
        size="md"
        variant="ghost"
        w={8}
        h={8}
      />
      <Menu isOpen={isOpen}>
        <MenuButton
          variant="ghost"
          as={IconButton}
          aria-label={t('stylePresets.templateActions')}
          tooltip={t('stylePresets.templateActions')}
          icon={<PiDotsThreeOutlineFill />}
          onClick={onToggle}
        />
        <MenuList>
          <StylePresetImport onClose={onClose} />
          <StylePresetExport onClose={onClose} />
        </MenuList>
      </Menu>
    </Flex>
  );
};
