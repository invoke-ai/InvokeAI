import { createSelector } from '@reduxjs/toolkit';
import { RootState } from 'app/store';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import { shiftKeyPressed } from 'features/ui/store/hotkeysSlice';
import { isEqual } from 'lodash';
import { isHotkeyPressed, useHotkeys } from 'react-hotkeys-hook';

const globalHotkeysSelector = createSelector(
  (state: RootState) => state.hotkeys,
  (hotkeys) => {
    const { shift } = hotkeys;
    return { shift };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: isEqual,
    },
  }
);

// TODO: Does not catch keypresses while focused in an input. Maybe there is a way?

export const useGlobalHotkeys = () => {
  const dispatch = useAppDispatch();
  const { shift } = useAppSelector(globalHotkeysSelector);

  useHotkeys(
    '*',
    () => {
      if (isHotkeyPressed('shift')) {
        !shift && dispatch(shiftKeyPressed(true));
      } else {
        shift && dispatch(shiftKeyPressed(false));
      }
    },
    { keyup: true, keydown: true },
    [shift]
  );
};
