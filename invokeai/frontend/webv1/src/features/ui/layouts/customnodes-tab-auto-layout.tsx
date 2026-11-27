import type { GridviewApi, IGridviewReactProps } from 'dockview';
import { GridviewReact, LayoutPriority, Orientation } from 'dockview';
import CustomNodesManagerTab from 'features/ui/components/tabs/CustomNodesManagerTab';
import type { RootLayoutGridviewComponents } from 'features/ui/layouts/auto-layout-context';
import { AutoLayoutProvider } from 'features/ui/layouts/auto-layout-context';
import type { TabName } from 'features/ui/store/uiTypes';
import { memo, useCallback, useEffect } from 'react';

import { navigationApi } from './navigation-api';
import { CUSTOM_NODES_PANEL_ID } from './shared';

const rootPanelComponents: RootLayoutGridviewComponents = {
  [CUSTOM_NODES_PANEL_ID]: CustomNodesManagerTab,
};

const initializeRootPanelLayout = (tab: TabName, api: GridviewApi) => {
  navigationApi.registerContainer(tab, 'root', api, () => {
    api.addPanel({
      id: CUSTOM_NODES_PANEL_ID,
      component: CUSTOM_NODES_PANEL_ID,
      priority: LayoutPriority.High,
    });
  });
};

export const CustomNodesTabAutoLayout = memo(() => {
  const onReady = useCallback<IGridviewReactProps['onReady']>(({ api }) => {
    initializeRootPanelLayout('customNodes', api);
  }, []);

  useEffect(
    () => () => {
      navigationApi.unregisterTab('customNodes');
    },
    []
  );

  return (
    <AutoLayoutProvider tab="customNodes">
      <GridviewReact
        className="dockview-theme-invoke"
        components={rootPanelComponents}
        onReady={onReady}
        orientation={Orientation.VERTICAL}
      />
    </AutoLayoutProvider>
  );
});
CustomNodesTabAutoLayout.displayName = 'CustomNodesTabAutoLayout';
