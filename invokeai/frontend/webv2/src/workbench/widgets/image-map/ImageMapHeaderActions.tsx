import type { WidgetViewProps } from '@workbench/widgetContracts';

import { ToggleIconButton } from '@platform/ui';
import { getImageMapClickSelectsCluster, getImageMapShowClusterLabels } from '@workbench/image-map/imageMapSettings';
import { useWidgetValuesSelector, useWorkbenchCommands } from '@workbench/WorkbenchContext';
import { GroupIcon, TagsIcon } from 'lucide-react';
import { useCallback } from 'react';

/**
 * Header toggles for the map's two click/display modes: whether a click selects
 * the whole cluster, and whether cluster labels are drawn. Both persist in the
 * widget's values, like the gallery's own view settings.
 */
export const ImageMapHeaderActions = (_props: WidgetViewProps) => {
  const { widgets } = useWorkbenchCommands();
  const clickSelectsCluster = useWidgetValuesSelector('image-map', getImageMapClickSelectsCluster);
  const showClusterLabels = useWidgetValuesSelector('image-map', getImageMapShowClusterLabels);
  const handleToggleClusterMode = useCallback(
    (checked: boolean) => widgets.patchValues('image-map', { clickSelectsCluster: checked }),
    [widgets]
  );
  const handleToggleLabels = useCallback(
    (checked: boolean) => widgets.patchValues('image-map', { showClusterLabels: checked }),
    [widgets]
  );

  return (
    <>
      <ToggleIconButton
        checked={showClusterLabels}
        icon={TagsIcon}
        label="Toggle cluster labels"
        tooltip={showClusterLabels ? 'Cluster labels shown' : 'Cluster labels hidden'}
        onCheckedChange={handleToggleLabels}
      />
      <ToggleIconButton
        checked={clickSelectsCluster}
        icon={GroupIcon}
        label="Toggle cluster selection mode"
        tooltip={clickSelectsCluster ? 'Click selects the whole cluster' : 'Click selects one image'}
        onCheckedChange={handleToggleClusterMode}
      />
    </>
  );
};
