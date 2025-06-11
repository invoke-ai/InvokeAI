import { useAppSelector } from 'app/store/storeHooks';
import type { GridviewApi, IGridviewReactProps } from 'dockview';
import { GridviewReact, Orientation } from 'dockview';
import { AutoLayoutProvider } from 'features/ui/layouts/auto-layout-context';
import { canvasTabComponents, initializeCanvasTabLayout } from 'features/ui/layouts/canvas-tab-auto-layout';
import { generateTabComponents, initializeGenerateTabLayout } from 'features/ui/layouts/generate-tab-auto-layout';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
import type { TabName } from 'features/ui/store/uiTypes';
import { memo, useCallback, useEffect, useState } from 'react';

const components: IGridviewReactProps['components'] = {
  ...generateTabComponents,
  ...canvasTabComponents,
};

export const AutoLayout = memo(() => {
  const tab = useAppSelector(selectActiveTab);
  const [api, setApi] = useState<GridviewApi | null>(null);
  const syncLayout = useCallback((tab: TabName, api: GridviewApi) => {
    if (tab === 'generate') {
      initializeGenerateTabLayout(api);
    } else if (tab === 'canvas') {
      initializeCanvasTabLayout(api);
    }
  }, []);
  const onReady = useCallback<IGridviewReactProps['onReady']>((event) => {
    setApi(event.api);
  }, []);
  useEffect(() => {
    if (api) {
      syncLayout(tab, api);
    }
  }, [api, syncLayout, tab]);
  return (
    <AutoLayoutProvider api={api}>
      <GridviewReact
        className="dockview-theme-invoke"
        components={components}
        onReady={onReady}
        orientation={Orientation.VERTICAL}
      />
    </AutoLayoutProvider>
  );
});
AutoLayout.displayName = 'AutoLayout';
