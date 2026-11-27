import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Flex, IconButton } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { $isStylePresetsMenuOpen } from 'features/stylePresets/store/stylePresetSlice';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCaretDownBold } from 'react-icons/pi';

import { ActiveStylePreset } from './ActiveStylePreset';

const _hover: SystemStyleObject = {
  bg: 'base.750',
};

export const StylePresetMenuTrigger = () => {
  const isMenuOpen = useStore($isStylePresetsMenuOpen);
  const { t } = useTranslation();

  const handleToggle = useCallback(() => {
    $isStylePresetsMenuOpen.set(!isMenuOpen);
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
      gap={2}
      role="button"
      _hover={_hover}
      transitionProperty="background-color"
      transitionDuration="normal"
      w="full"
    >
      <ActiveStylePreset />
      <IconButton aria-label={t('stylePresets.viewList')} variant="ghost" icon={<PiCaretDownBold />} size="sm" />
    </Flex>
  );
};
