import type { DockviewApi, GridviewApi, IDockviewReactProps, IGridviewReactProps } from 'dockview';
import { DockviewReact, GridviewReact, LayoutPriority, Orientation } from 'dockview';
import { BottomGalleryPanel } from 'features/gallery/components/BottomGalleryPanel';
import { ImageViewerPanel } from 'features/gallery/components/ImageViewer/ImageViewerPanel';
import { FloatingLeftPanelButtons } from 'features/ui/components/FloatingLeftPanelButtons';
import type {
  AutoLayoutDockviewComponents,
  AutoLayoutGridviewComponents,
  DockviewPanelParameters,
  GridviewPanelParameters,
  RootLayoutGridviewComponents,
} from 'features/ui/layouts/auto-layout-context';
import { AutoLayoutProvider, useAutoLayoutContext, withPanelContainer } from 'features/ui/layouts/auto-layout-context';
import type { TabName } from 'features/ui/store/uiTypes';
import { dockviewTheme } from 'features/ui/styles/theme';
import { t } from 'i18next';
import { memo, useCallback, useEffect } from 'react';

import { DockviewTab } from './DockviewTab';
import { DockviewTabLaunchpad } from './DockviewTabLaunchpad';
import { DockviewTabProgress } from './DockviewTabProgress';
import { GenerateLaunchpadPanel } from './GenerateLaunchpadPanel';
import { GenerateTabLeftPanel } from './GenerateTabLeftPanel';
import { navigationApi } from './navigation-api';
import { PanelHotkeysLogical } from './PanelHotkeysLogical';
import {
  BOTTOM_GALLERY_DEFAULT_HEIGHT_PX,
  BOTTOM_GALLERY_MIN_HEIGHT_PX,
  BOTTOM_GALLERY_PANEL_ID,
  DOCKVIEW_TAB_ID,
  DOCKVIEW_TAB_LAUNCHPAD_ID,
  DOCKVIEW_TAB_PROGRESS_ID,
  LAUNCHPAD_PANEL_ID,
  LEFT_PANEL_ID,
  LEFT_PANEL_MIN_SIZE_PX,
  MAIN_PANEL_ID,
  SETTINGS_PANEL_ID,
  TOP_AREA_ID,
  VIEWER_PANEL_ID,
} from './shared';

const tabComponents = {
  [DOCKVIEW_TAB_ID]: DockviewTab,
  [DOCKVIEW_TAB_PROGRESS_ID]: DockviewTabProgress,
  [DOCKVIEW_TAB_LAUNCHPAD_ID]: DockviewTabLaunchpad,
};

const mainPanelComponents: AutoLayoutDockviewComponents = {
  [LAUNCHPAD_PANEL_ID]: withPanelContainer(GenerateLaunchpadPanel),
  [VIEWER_PANEL_ID]: withPanelContainer(ImageViewerPanel),
};

const initializeMainPanelLayout = (tab: TabName, api: DockviewApi) => {
  navigationApi.registerContainer(tab, 'main', api, () => {
    const launchpad = api.addPanel<DockviewPanelParameters>({
      id: LAUNCHPAD_PANEL_ID,
      component: LAUNCHPAD_PANEL_ID,
      title: t('ui.panels.launchpad'),
      tabComponent: DOCKVIEW_TAB_LAUNCHPAD_ID,
      params: {
        tab,
        focusRegion: 'launchpad',
        i18nKey: 'ui.panels.launchpad',
      },
    });

    api.addPanel<DockviewPanelParameters>({
      id: VIEWER_PANEL_ID,
      component: VIEWER_PANEL_ID,
      title: t('ui.panels.imageViewer'),
      tabComponent: DOCKVIEW_TAB_PROGRESS_ID,
      params: {
        tab,
        focusRegion: 'viewer',
        i18nKey: 'ui.panels.imageViewer',
      },
      position: {
        direction: 'within',
        referencePanel: launchpad.id,
      },
    });

    launchpad.api.setActive();
  });
};

const MainPanel = memo(() => {
  const { tab } = useAutoLayoutContext();

  const onReady = useCallback<IDockviewReactProps['onReady']>(
    ({ api }) => {
      initializeMainPanelLayout(tab, api);
    },
    [tab]
  );
  return (
    <>
      <DockviewReact
        disableDnd={true}
        locked={true}
        disableFloatingGroups={true}
        dndEdges={false}
        tabComponents={tabComponents}
        components={mainPanelComponents}
        onReady={onReady}
        theme={dockviewTheme}
      />
      <FloatingLeftPanelButtons />
      <PanelHotkeysLogical />
    </>
  );
});
MainPanel.displayName = 'MainPanel';

const leftPanelComponents: AutoLayoutGridviewComponents = {
  [SETTINGS_PANEL_ID]: withPanelContainer(GenerateTabLeftPanel),
};

const initializeLeftPanelLayout = (tab: TabName, api: GridviewApi) => {
  navigationApi.registerContainer(tab, 'left', api, () => {
    api.addPanel<GridviewPanelParameters>({
      id: SETTINGS_PANEL_ID,
      component: SETTINGS_PANEL_ID,
      params: {
        tab,
        focusRegion: 'settings',
      },
    });
  });
};

const LeftPanel = memo(() => {
  const { tab } = useAutoLayoutContext();

  const onReady = useCallback<IGridviewReactProps['onReady']>(
    ({ api }) => {
      initializeLeftPanelLayout(tab, api);
    },
    [tab]
  );
  return (
    <GridviewReact
      className="dockview-theme-invoke"
      orientation={Orientation.VERTICAL}
      components={leftPanelComponents}
      onReady={onReady}
    />
  );
});
LeftPanel.displayName = 'LeftPanel';

/**
 * The top area contains the left panel and main panel side by side.
 * This is a nested GridviewReact with VERTICAL orientation (which arranges panels horizontally).
 */
const topAreaComponents: RootLayoutGridviewComponents = {
  [LEFT_PANEL_ID]: LeftPanel,
  [MAIN_PANEL_ID]: MainPanel,
};

const initializeTopAreaLayout = (tab: TabName, api: GridviewApi) => {
  navigationApi.registerContainer(tab, 'top-area', api, () => {
    const main = api.addPanel<GridviewPanelParameters>({
      id: MAIN_PANEL_ID,
      component: MAIN_PANEL_ID,
      priority: LayoutPriority.High,
    });

    const left = api.addPanel<GridviewPanelParameters>({
      id: LEFT_PANEL_ID,
      component: LEFT_PANEL_ID,
      minimumWidth: LEFT_PANEL_MIN_SIZE_PX,
      position: {
        direction: 'left',
        referencePanel: main.id,
      },
    });

    left.api.setSize({ width: LEFT_PANEL_MIN_SIZE_PX });
  });
};

const TopArea = memo(() => {
  const { tab } = useAutoLayoutContext();

  const onReady = useCallback<IGridviewReactProps['onReady']>(
    ({ api }) => {
      initializeTopAreaLayout(tab, api);
    },
    [tab]
  );
  return (
    <GridviewReact
      className="dockview-theme-invoke"
      components={topAreaComponents}
      onReady={onReady}
      orientation={Orientation.VERTICAL}
    />
  );
});
TopArea.displayName = 'TopArea';

/**
 * Bottom gallery panel wrapper for use in GridviewReact.
 */
const bottomGalleryComponents: AutoLayoutGridviewComponents = {
  [BOTTOM_GALLERY_PANEL_ID]: withPanelContainer(BottomGalleryPanel),
};

/**
 * Root layout: vertical split with top area and bottom gallery.
 * Orientation.HORIZONTAL arranges panels vertically (top/bottom).
 */
const rootPanelComponents: RootLayoutGridviewComponents = {
  [TOP_AREA_ID]: TopArea,
  [BOTTOM_GALLERY_PANEL_ID]: bottomGalleryComponents[BOTTOM_GALLERY_PANEL_ID]!,
};

const initializeRootPanelLayout = (tab: TabName, api: GridviewApi) => {
  navigationApi.registerContainer(tab, 'root', api, () => {
    const topArea = api.addPanel<GridviewPanelParameters>({
      id: TOP_AREA_ID,
      component: TOP_AREA_ID,
      priority: LayoutPriority.High,
    });

    const bottomGallery = api.addPanel<GridviewPanelParameters>({
      id: BOTTOM_GALLERY_PANEL_ID,
      component: BOTTOM_GALLERY_PANEL_ID,
      minimumHeight: BOTTOM_GALLERY_MIN_HEIGHT_PX,
      params: {
        tab,
        focusRegion: 'gallery',
      },
      position: {
        direction: 'below',
        referencePanel: topArea.id,
      },
    });

    bottomGallery.api.setSize({ height: BOTTOM_GALLERY_DEFAULT_HEIGHT_PX });
  });
};

export const GenerateTabAutoLayout = memo(() => {
  const onReady = useCallback<IGridviewReactProps['onReady']>(({ api }) => {
    initializeRootPanelLayout('generate', api);
  }, []);

  useEffect(
    () => () => {
      navigationApi.unregisterTab('generate');
    },
    []
  );

  return (
    <AutoLayoutProvider tab="generate">
      <GridviewReact
        className="dockview-theme-invoke"
        components={rootPanelComponents}
        onReady={onReady}
        orientation={Orientation.HORIZONTAL}
      />
    </AutoLayoutProvider>
  );
});
GenerateTabAutoLayout.displayName = 'GenerateTabAutoLayout';
