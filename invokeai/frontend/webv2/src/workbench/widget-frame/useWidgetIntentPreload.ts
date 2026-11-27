import type { RegisteredWidget } from '@workbench/widgetContracts';

import { useCallback } from 'react';

export const useWidgetIntentPreloadProps = (widget: RegisteredWidget, disabled = false) => {
  const handleIntent = useCallback(() => {
    if (!disabled && widget.status === 'enabled') {
      widget.implementation.preload();
    }
  }, [disabled, widget.implementation, widget.status]);

  return { onFocus: handleIntent, onPointerEnter: handleIntent };
};
