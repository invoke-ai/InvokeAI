import type { WidgetViewProps } from '@workbench/widgetContracts';

import { ToggleIconButton } from '@platform/ui';
import { getImageMapClickSelectsCluster } from '@workbench/image-map/imageMapSettings';
import { useWidgetValuesSelector, useWorkbenchCommands } from '@workbench/WorkbenchContext';
import { GroupIcon } from 'lucide-react';
import { useCallback } from 'react';

/**
 * Header toggle for cluster-selection mode: when on, clicking a map point
 * selects its whole cluster in the gallery instead of just the one image.
 * Persisted in the widget's values, like the gallery's own view settings.
 */
export const ImageMapHeaderActions = (_props: WidgetViewProps) => {
  const { widgets } = useWorkbenchCommands();
  const clickSelectsCluster = useWidgetValuesSelector('image-map', getImageMapClickSelectsCluster);
  const handleToggle = useCallback(
    (checked: boolean) => widgets.patchValues('image-map', { clickSelectsCluster: checked }),
    [widgets]
  );

  return (
    <ToggleIconButton
      checked={clickSelectsCluster}
      icon={GroupIcon}
      label="Toggle cluster selection mode"
      tooltip={clickSelectsCluster ? 'Click selects the whole cluster' : 'Click selects one image'}
      onCheckedChange={handleToggle}
    />
  );
};
