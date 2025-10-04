import type { GridviewApi, IGridviewReactProps } from 'dockview';
import { GridviewReact, LayoutPriority, Orientation } from 'dockview';
import type { TabName } from 'features/controlLayers/store/types';
import QueueTab from 'features/ui/components/tabs/QueueTab';
import type { RootLayoutGridviewComponents } from 'features/ui/layouts/auto-layout-context';
import { AutoLayoutProvider } from 'features/ui/layouts/auto-layout-context';
import { memo, useCallback, useEffect } from 'react';

import { navigationApi } from './navigation-api';
import { QUEUE_PANEL_ID } from './shared';

const rootPanelComponents: RootLayoutGridviewComponents = {
  [QUEUE_PANEL_ID]: QueueTab,
};

const initializeRootPanelLayout = (tab: TabName, api: GridviewApi) => {
  navigationApi.registerContainer(tab, 'root', api, () => {
    api.addPanel({
      id: QUEUE_PANEL_ID,
      component: QUEUE_PANEL_ID,
      priority: LayoutPriority.High,
    });
  });
};

export const QueueTabAutoLayout = memo(() => {
  const onReady = useCallback<IGridviewReactProps['onReady']>(({ api }) => {
    initializeRootPanelLayout('queue', api);
  }, []);

  useEffect(
    () => () => {
      navigationApi.unregisterTab('queue');
    },
    []
  );

  return (
    <AutoLayoutProvider tab="queue">
      <GridviewReact
        className="dockview-theme-invoke"
        components={rootPanelComponents}
        onReady={onReady}
        orientation={Orientation.VERTICAL}
      />
    </AutoLayoutProvider>
  );
});
QueueTabAutoLayout.displayName = 'QueueTabAutoLayout';
