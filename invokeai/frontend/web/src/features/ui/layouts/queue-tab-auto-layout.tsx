import type { GridviewApi, IGridviewPanel, IGridviewReactProps } from 'dockview';
import { GridviewReact, LayoutPriority, Orientation } from 'dockview';
import QueueTab from 'features/ui/components/tabs/QueueTab';
import type { RootLayoutGridviewComponents } from 'features/ui/layouts/auto-layout-context';
import { AutoLayoutProvider } from 'features/ui/layouts/auto-layout-context';
import { memo, useCallback, useEffect } from 'react';

import { navigationApi } from './navigation-api';
import { QUEUE_PANEL_ID } from './shared';

const rootPanelComponents: RootLayoutGridviewComponents = {
  [QUEUE_PANEL_ID]: QueueTab,
};

const initializeRootPanelLayout = (layoutApi: GridviewApi) => {
  const queue = layoutApi.addPanel({
    id: QUEUE_PANEL_ID,
    component: QUEUE_PANEL_ID,
    priority: LayoutPriority.High,
  });

  navigationApi.registerPanel('queue', QUEUE_PANEL_ID, queue);

  return { queue } satisfies Record<string, IGridviewPanel>;
};

export const QueueTabAutoLayout = memo(() => {
  const onReady = useCallback<IGridviewReactProps['onReady']>(({ api }) => {
    initializeRootPanelLayout(api);
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
