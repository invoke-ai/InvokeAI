import { useCallback } from 'react';

import { useWorkbench } from '../../WorkbenchContext';

/**
 * Bring the full model manager into the center area, enabling it there first
 * for projects persisted before the widget existed.
 */
export const useOpenModelManager = () => {
  const { activeProject, dispatch } = useWorkbench();
  const isEnabledInCenter = activeProject.widgetRegions.center.enabledWidgetIds.includes('models');

  return useCallback(() => {
    if (!isEnabledInCenter) {
      dispatch({ region: 'center', type: 'toggleRegionWidget', widgetId: 'models' });
    }

    dispatch({ region: 'center', type: 'selectRegionWidget', widgetId: 'models' });
  }, [dispatch, isEnabledInCenter]);
};
