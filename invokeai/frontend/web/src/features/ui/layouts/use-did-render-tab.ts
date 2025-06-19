import { addAppListener } from 'app/store/middleware/listenerMiddleware';
import { useAppDispatch } from 'app/store/storeHooks';
import { setActiveTab } from 'features/ui/store/uiSlice';
import type { TabName } from 'features/ui/store/uiTypes';
import { useEffect } from 'react';

export const useOnFirstVisitToTab = (tab: TabName, cb: () => void) => {
  const dispatch = useAppDispatch();
  useEffect(() => {
    dispatch(
      addAppListener({
        predicate: (action) => {
          if (!setActiveTab.match(action)) {
            return false;
          }
          return action.payload === tab;
        },
        effect: (_, api) => {
          cb();
          api.unsubscribe();
        },
      })
    );
  }, [cb, dispatch, tab]);
};
