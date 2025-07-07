import type { GridviewApi, IGridviewPanel, IGridviewReactProps } from 'dockview';
import { GridviewReact, LayoutPriority, Orientation } from 'dockview';
import ModelManagerTab from 'features/ui/components/tabs/ModelManagerTab';
import type { RootLayoutGridviewComponents } from 'features/ui/layouts/auto-layout-context';
import { AutoLayoutProvider } from 'features/ui/layouts/auto-layout-context';
import { memo, useCallback, useEffect } from 'react';

import { navigationApi } from './navigation-api';
import { MODELS_PANEL_ID } from './shared';

const rootPanelComponents: RootLayoutGridviewComponents = {
  [MODELS_PANEL_ID]: ModelManagerTab,
};

const initializeRootPanelLayout = (layoutApi: GridviewApi) => {
  const models = layoutApi.addPanel({
    id: MODELS_PANEL_ID,
    component: MODELS_PANEL_ID,
    priority: LayoutPriority.High,
  });

  navigationApi.registerPanel('models', MODELS_PANEL_ID, models);

  return { models } satisfies Record<string, IGridviewPanel>;
};

export const ModelsTabAutoLayout = memo(() => {
  const onReady = useCallback<IGridviewReactProps['onReady']>(({ api }) => {
    initializeRootPanelLayout(api);
    navigationApi.onTabReady('models');
  }, []);

  useEffect(
    () => () => {
      navigationApi.unregisterTab('models');
    },
    []
  );

  return (
    <AutoLayoutProvider tab="models">
      <GridviewReact
        className="dockview-theme-invoke"
        components={rootPanelComponents}
        onReady={onReady}
        orientation={Orientation.VERTICAL}
      />
    </AutoLayoutProvider>
  );
});
ModelsTabAutoLayout.displayName = 'ModelsTabAutoLayout';
