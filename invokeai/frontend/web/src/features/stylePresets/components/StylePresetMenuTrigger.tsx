import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Flex, IconButton } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { usePromptExpansionTracking } from 'features/prompt/PromptExpansion/usePromptExpansionTracking';
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
  const { isPending: isPromptExpansionPending } = usePromptExpansionTracking();
  const { t } = useTranslation();

  const handleToggle = useCallback(() => {
    if (isPromptExpansionPending) {
      return;
    }
    $isStylePresetsMenuOpen.set(!isMenuOpen);
  }, [isMenuOpen, isPromptExpansionPending]);

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
      _hover={isPromptExpansionPending ? undefined : _hover}
      transitionProperty="background-color"
      transitionDuration="normal"
      w="full"
      opacity={isPromptExpansionPending ? 0.5 : 1}
      cursor={isPromptExpansionPending ? 'not-allowed' : 'pointer'}
    >
      <ActiveStylePreset />
      <IconButton
        aria-label={t('stylePresets.viewList')}
        variant="ghost"
        icon={<PiCaretDownBold />}
        size="sm"
        isDisabled={isPromptExpansionPending}
      />
    </Flex>
  );
};
