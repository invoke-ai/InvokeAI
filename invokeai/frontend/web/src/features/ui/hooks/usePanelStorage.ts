import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useCallback } from 'react';
import { panelsChanged } from '../store/uiSlice';

export const usePanelStorage = () => {
  const dispatch = useAppDispatch();
  const panels = useAppSelector((state) => state.ui.panels);
  const getItem = useCallback((name: string) => panels[name] ?? '', [panels]);
  const setItem = useCallback(
    (name: string, value: string) => {
      dispatch(panelsChanged({ name, value }));
    },
    [dispatch]
  );

  return { getItem, setItem };
};
