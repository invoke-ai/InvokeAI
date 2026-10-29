import type { GalleryMediaFilter } from '@features/gallery/contracts';
import type { WidgetViewProps } from '@workbench/widgetContracts';

import { getGalleryMediaFilter, getGalleryMediaFilterMenuProps } from '@features/gallery/contracts';
import { ToggleIconButton } from '@platform/ui';
import { IconRadioMenu } from '@platform/ui/Menu';
import {
  getImageMapClickSelectsCluster,
  getImageMapShowClusterLabels,
  IMAGE_MAP_MEDIA_FILTER_TRIGGER_ID,
} from '@workbench/image-map/imageMapSettings';
import { useWidgetValuesSelector, useWorkbenchCommands } from '@workbench/WorkbenchContext';
import { GroupIcon, TagsIcon } from 'lucide-react';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export const ImageMapHeaderActions = (_props: WidgetViewProps) => {
  const { t } = useTranslation();
  const { gallery, widgets } = useWorkbenchCommands();
  // The Gallery's per-project setting: the map shows what the Gallery lists.
  const mediaFilter = useWidgetValuesSelector('gallery', getGalleryMediaFilter);
  const clickSelectsCluster = useWidgetValuesSelector('image-map', getImageMapClickSelectsCluster);
  const showClusterLabels = useWidgetValuesSelector('image-map', getImageMapShowClusterLabels);
  const handleToggleClusterMode = useCallback(
    (checked: boolean) => widgets.patchValues('image-map', { clickSelectsCluster: checked }),
    [widgets]
  );
  const handleMediaFilterChange = useCallback(
    (nextMediaFilter: GalleryMediaFilter) => gallery.updateSettings({ mediaFilter: nextMediaFilter }),
    [gallery]
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
        tooltip={
          clickSelectsCluster ? 'Click shows the whole cluster in the gallery' : 'Click reveals one item in the gallery'
        }
        onCheckedChange={handleToggleClusterMode}
      />
      {/* Last, so it sits directly beside the settings button the frame adds after the header actions. */}
      <IconRadioMenu
        {...getGalleryMediaFilterMenuProps(t, mediaFilter)}
        triggerId={IMAGE_MAP_MEDIA_FILTER_TRIGGER_ID}
        value={mediaFilter}
        onValueChange={handleMediaFilterChange}
      />
    </>
  );
};
