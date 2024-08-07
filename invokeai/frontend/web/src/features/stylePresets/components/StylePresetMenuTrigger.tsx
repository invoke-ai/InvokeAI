import { Flex, Icon } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { isMenuOpenChanged } from 'features/stylePresets/store/stylePresetSlice';
import { useCallback } from 'react';
import { PiCaretDownBold } from 'react-icons/pi';

import { ActiveStylePreset } from './ActiveStylePreset';

export const StylePresetMenuTrigger = () => {
  const isMenuOpen = useAppSelector((s) => s.stylePreset.isMenuOpen);
  const dispatch = useAppDispatch();

  const handleToggle = useCallback(() => {
    dispatch(isMenuOpenChanged(!isMenuOpen));
  }, [dispatch, isMenuOpen]);

  return (
    <Flex
      as="button"
      onClick={handleToggle}
      backgroundColor="base.800"
      justifyContent="space-between"
      alignItems="center"
      padding="5px 10px"
      borderRadius="base"
      gap="2"
    >
      <ActiveStylePreset />

      <Icon boxSize="15px" as={PiCaretDownBold} color="base.300" />
    </Flex>
  );
};
