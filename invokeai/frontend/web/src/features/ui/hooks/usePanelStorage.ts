import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { panelsChanged } from 'features/ui/store/uiSlice';
import { useCallback } from 'react';

export const usePanelStorage = () => {
  const dispatch = useAppDispatch();
  const panels = useAppSelector((s) => s.ui.panels);
  const getItem = useCallback((name: string) => panels[name] ?? '', [panels]);
  const setItem = useCallback(
    (name: string, value: string) => {
      dispatch(panelsChanged({ name, value }));
    },
    [dispatch]
  );

  return { getItem, setItem };
};
