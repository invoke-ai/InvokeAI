import type { TabName } from 'features/controlLayers/store/types';

import {
  GALLERY_PANEL_DEFAULT_HEIGHT_PX,
  GALLERY_PANEL_ID,
  GALLERY_PANEL_MIN_EXPANDED_HEIGHT_PX,
  GALLERY_PANEL_MIN_HEIGHT_PX,
} from './shared';
import { useCollapsibleGridviewPanel } from './use-collapsible-gridview-panel';

export const useGalleryPanel = (tab: TabName) => {
  return useCollapsibleGridviewPanel(
    tab,
    GALLERY_PANEL_ID,
    'vertical',
    GALLERY_PANEL_DEFAULT_HEIGHT_PX,
    GALLERY_PANEL_MIN_HEIGHT_PX,
    GALLERY_PANEL_MIN_EXPANDED_HEIGHT_PX
  );
};
