import { useAppSelector } from 'app/store/storeHooks';
import { useIsRegionFocused } from 'common/hooks/focus';
import { selectHasMetadataViewerItem } from 'features/gallery/store/gallerySelectors';
import { getSharedInfoHotkeyTarget } from 'features/system/components/HotkeysModal/sharedHotkeyRouting';
import { navigationApi } from 'features/ui/layouts/navigation-api';
import { VIEWER_PANEL_ID } from 'features/ui/layouts/shared';
import { selectActiveTab } from 'features/ui/store/uiSelectors';

export const useSharedInfoHotkeyTarget = () => {
  const activeTab = useAppSelector(selectActiveTab);
  const hasMetadataViewerItem = useAppSelector(selectHasMetadataViewerItem);
  const isGalleryFocused = useIsRegionFocused('gallery');
  const isViewerFocused = useIsRegionFocused('viewer');
  const isViewerPanelActive = navigationApi.isDockviewPanelActive(activeTab, VIEWER_PANEL_ID);

  return getSharedInfoHotkeyTarget({
    hasMetadataViewerItem,
    isGalleryFocused,
    isViewerPanelActive,
    isViewerFocused,
  });
};
