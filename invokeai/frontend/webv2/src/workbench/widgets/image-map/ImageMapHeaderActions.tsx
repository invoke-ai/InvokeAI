import type { WidgetViewProps } from '@workbench/widgetContracts';

import { Icon } from '@chakra-ui/react';
import { IconButton, Tooltip } from '@platform/ui';
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
  const clickSelectsCluster = useWidgetValuesSelector('image-map', (values) => Boolean(values.clickSelectsCluster));
  const handleToggle = useCallback(
    () => widgets.patchValues('image-map', { clickSelectsCluster: !clickSelectsCluster }),
    [clickSelectsCluster, widgets]
  );

  return (
    <Tooltip content={clickSelectsCluster ? 'Click selects the whole cluster' : 'Click selects one image'}>
      <IconButton
        aria-label="Toggle cluster selection mode"
        aria-pressed={clickSelectsCluster}
        color={clickSelectsCluster ? 'accent.fg' : 'fg.muted'}
        size="2xs"
        variant={clickSelectsCluster ? 'subtle' : 'ghost'}
        onClick={handleToggle}
      >
        <Icon as={GroupIcon} boxSize="3.5" />
      </IconButton>
    </Tooltip>
  );
};
