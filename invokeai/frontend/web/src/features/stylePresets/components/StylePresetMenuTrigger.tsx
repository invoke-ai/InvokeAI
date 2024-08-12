import { Flex, IconButton } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { $isMenuOpen } from 'features/stylePresets/store/isMenuOpen';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCaretDownBold } from 'react-icons/pi';

import { ActiveStylePreset } from './ActiveStylePreset';

export const StylePresetMenuTrigger = () => {
  const isMenuOpen = useStore($isMenuOpen);
  const { t } = useTranslation();

  const handleToggle = useCallback(() => {
    $isMenuOpen.set(!isMenuOpen);
  }, [isMenuOpen]);

  return (
    <Flex
      onClick={handleToggle}
      backgroundColor="base.800"
      justifyContent="space-between"
      alignItems="center"
      py={2}
      px={3}
      borderRadius="base"
      gap={1}
      role="button"
    >
      <ActiveStylePreset />

      <IconButton aria-label={t('stylePresets.viewList')} variant="ghost" icon={<PiCaretDownBold />} size="sm" />
    </Flex>
  );
};
