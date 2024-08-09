import { Flex, IconButton } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { isMenuOpenChanged } from 'features/stylePresets/store/stylePresetSlice';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCaretDownBold } from 'react-icons/pi';

import { ActiveStylePreset } from './ActiveStylePreset';

export const StylePresetMenuTrigger = () => {
  const isMenuOpen = useAppSelector((s) => s.stylePreset.isMenuOpen);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const handleToggle = useCallback(() => {
    dispatch(isMenuOpenChanged(!isMenuOpen));
  }, [dispatch, isMenuOpen]);

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
