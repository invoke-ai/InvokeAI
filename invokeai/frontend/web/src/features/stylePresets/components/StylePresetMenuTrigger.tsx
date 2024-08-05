import { Button, Popover, PopoverBody, PopoverContent, PopoverTrigger } from '@invoke-ai/ui-library';

import { StylePresetMenu } from './StylePresetMenu';
import { useAppDispatch, useAppSelector } from '../../../app/store/storeHooks';
import { useCallback } from 'react';
import { isMenuOpenChanged } from '../store/stylePresetSlice';

export const StylePresetMenuTrigger = () => {
  const isMenuOpen = useAppSelector((s) => s.stylePreset.isMenuOpen);
  const dispatch = useAppDispatch();

  const handleClose = useCallback(() => {
    dispatch(isMenuOpenChanged(false));
  }, [dispatch]);

  const handleToggle = useCallback(() => {
    dispatch(isMenuOpenChanged(!isMenuOpen));
  }, [dispatch, isMenuOpen]);

  return (
    <Popover isOpen={isMenuOpen} onClose={handleClose}>
      <PopoverTrigger>
        <Button size="sm" onClick={handleToggle}>
          Style Presets
        </Button>
      </PopoverTrigger>
      <PopoverContent>
        <PopoverBody>
          <StylePresetMenu />
        </PopoverBody>
      </PopoverContent>
    </Popover>
  );
};
