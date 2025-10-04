import type { GridviewApi, IGridviewReactProps } from 'dockview';
import { GridviewReact, LayoutPriority, Orientation } from 'dockview';
import type { TabName } from 'features/controlLayers/store/types';
import ModelManagerTab from 'features/ui/components/tabs/ModelManagerTab';
import type { RootLayoutGridviewComponents } from 'features/ui/layouts/auto-layout-context';
import { AutoLayoutProvider } from 'features/ui/layouts/auto-layout-context';
import { memo, useCallback, useEffect } from 'react';

import { navigationApi } from './navigation-api';
import { MODELS_PANEL_ID } from './shared';

const rootPanelComponents: RootLayoutGridviewComponents = {
  [MODELS_PANEL_ID]: ModelManagerTab,
};

const initializeRootPanelLayout = (tab: TabName, api: GridviewApi) => {
  navigationApi.registerContainer(tab, 'root', api, () => {
    api.addPanel({
      id: MODELS_PANEL_ID,
      component: MODELS_PANEL_ID,
      priority: LayoutPriority.High,
    });
  });
};

export const ModelsTabAutoLayout = memo(() => {
  const onReady = useCallback<IGridviewReactProps['onReady']>(({ api }) => {
    initializeRootPanelLayout('models', api);
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
