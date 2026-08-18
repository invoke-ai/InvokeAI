import { writeBootWidgetHint } from '@workbench/bootWidgetPreload';
import { getLayoutWidgetTypeIds } from '@workbench/layoutWidgetSet';
import { shallowEqual, useActiveProjectSelector } from '@workbench/WorkbenchContext';
import { useEffect } from 'react';

/**
 * Persists the active layout's widget set so the next boot can start those
 * chunk downloads in parallel with project hydration (`preloadBootWidgets`).
 * Mirrors the theme hint in `ThemeController`: a dedicated key, written after
 * the store has the real layout, read before anyone knows who is signed in.
 * Renders nothing.
 */
export const BootWidgetHintController = () => {
  const typeIds = useActiveProjectSelector(getLayoutWidgetTypeIds, shallowEqual);

  useEffect(() => {
    writeBootWidgetHint(typeIds);
  }, [typeIds]);

  return null;
};
