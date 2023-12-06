import { useAppSelector } from 'app/store/storeHooks';
import { useEffect } from 'react';

/**
 * The reactflow background element somehow prevents the chakra `useOutsideClick()` hook from working.
 * With a menu open, clicking on the reactflow background element doesn't close the menu.
 *
 * Reactflow does provide an `onPaneClick` to handle clicks on the background element, but it is not
 * straightforward to programatically close the menu.
 *
 * As a (hopefully temporary) workaround, we will use a dirty hack:
 * - create `globalMenuCloseTrigger: number` in `ui` slice
 * - increment it in `onPaneClick`
 * - `useEffect()` to close the menu when `globalMenuCloseTrigger` changes
 */

export const useGlobalMenuCloseTrigger = (onClose: () => void) => {
  const globalMenuCloseTrigger = useAppSelector(
    (state) => state.ui.globalMenuCloseTrigger
  );

  useEffect(() => {
    onClose();
  }, [globalMenuCloseTrigger, onClose]);
};
