import { useAppDispatch } from 'app/store/storeHooks';
import type { InvokeTabName } from 'features/ui/store/tabMap';
import { setActiveTab } from 'features/ui/store/uiSlice';
import { useCallback, useEffect } from 'react';

export const useDestination = (destination: InvokeTabName | undefined) => {
  const dispatch = useAppDispatch();

  const handleSendToDestination = useCallback(() => {
    if (destination) {
      dispatch(setActiveTab(destination));
    }
  }, [dispatch, destination]);

  useEffect(() => {
    handleSendToDestination();
  }, [destination, handleSendToDestination]);

  return { handleSendToDestination };
};
