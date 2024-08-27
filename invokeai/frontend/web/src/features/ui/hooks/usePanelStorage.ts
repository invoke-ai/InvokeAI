import { useAppStore } from 'app/store/nanostores/store';
import { useAppDispatch } from 'app/store/storeHooks';
import { panelsChanged } from 'features/ui/store/uiSlice';
import { useCallback } from 'react';

export const usePanelStorage = () => {
  const store = useAppStore();
  const dispatch = useAppDispatch();
  const getItem = useCallback(
    (name: string) => {
      const panels = store.getState().ui.panels;
      return panels[name] ?? '';
    },
    [store]
  );
  const setItem = useCallback(
    (name: string, value: string) => {
      dispatch(panelsChanged({ name, value }));
    },
    [dispatch]
  );

  return { getItem, setItem };
};
