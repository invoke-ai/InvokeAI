import { writeBootWidgetHint } from '@workbench/bootWidgetPreload';
import { getLayoutWidgetTypeIds } from '@workbench/layoutWidgetSet';
import { shallowEqual, useActiveProjectSelector } from '@workbench/WorkbenchContext';
import { useEffect } from 'react';

/** Persist resolved widget hints for pre-hydration chunk loading on the next boot, before identity is known. */
export const BootWidgetHintController = () => {
  const typeIds = useActiveProjectSelector(getLayoutWidgetTypeIds, shallowEqual);

  useEffect(() => {
    writeBootWidgetHint(typeIds);
  }, [typeIds]);

  return null;
};
