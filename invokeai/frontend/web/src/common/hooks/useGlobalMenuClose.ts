import { atom } from 'nanostores';
import { useCallback, useEffect } from 'react';

type CB = () => void;

const $onCloseCallbacks = atom<CB[]>([]);

/**
 * The reactflow background element somehow prevents the chakra `useOutsideClick()` hook from working.
 * With a menu open, clicking on the reactflow background element doesn't close the menu.
 *
 * Reactflow does provide an `onPaneClick` to handle clicks on the background element, but it is not
 * straightforward to programatically close all menus.
 *
 * This hook provides a way to close all menus by calling `onCloseGlobal()`. Menus that want to be closed
 * in this way should register themselves by passing a callback to `useGlobalMenuCloseTrigger()`.
 */
export const useGlobalMenuClose = (onClose?: CB) => {
  useEffect(() => {
    if (!onClose) {
      return;
    }
    $onCloseCallbacks.set([...$onCloseCallbacks.get(), onClose]);
    return () => {
      $onCloseCallbacks.set(
        $onCloseCallbacks.get().filter((c) => c !== onClose)
      );
    };
  }, [onClose]);

  const onCloseGlobal = useCallback(() => {
    $onCloseCallbacks.get().forEach((cb) => cb());
  }, []);

  return { onCloseGlobal };
};
