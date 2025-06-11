import { Box, Divider, Flex } from '@invoke-ai/ui-library';
import type { GridviewApi, IDockviewReactProps, IGridviewReactProps } from 'dockview';
import { DockviewReact, GridviewReact, Orientation } from 'dockview';
import { InitialState } from 'features/controlLayers/components/SimpleSession/InitialState';
import { BoardsListPanelContent } from 'features/gallery/components/BoardsListPanelContent';
import { Gallery } from 'features/gallery/components/Gallery';
import { ImageViewer } from 'features/gallery/components/ImageViewer/ImageViewer2';
import { ProgressImage } from 'features/gallery/components/ImageViewer/ProgressImage2';
import { ViewerToolbar } from 'features/gallery/components/ImageViewer/ViewerToolbar2';
import QueueControls from 'features/queue/components/QueueControls';
import ParametersPanelTextToImage from 'features/ui/components/ParametersPanels/ParametersPanelTextToImage';
import { AutoLayoutProvider } from 'features/ui/layouts/auto-layout-context';
import { TabWithoutCloseButton } from 'features/ui/layouts/TabWithoutCloseButton';
import { LEFT_PANEL_MIN_SIZE_PX, RIGHT_PANEL_MIN_SIZE_PX } from 'features/ui/store/uiSlice';
import { dockviewTheme } from 'features/ui/styles/theme';
import { memo, useCallback, useState } from 'react';

const ViewerPanelContent = memo(() => (
  <Flex flexDir="column" w="full" h="full" overflow="hidden" p={2} gap={2}>
    <ViewerToolbar />
    <Divider />
    <ImageViewer />
  </Flex>
));
ViewerPanelContent.displayName = 'ViewerPanelContent';

const ProgressPanelContent = memo(() => (
  <Flex flexDir="column" w="full" h="full" overflow="hidden" p={2}>
    <ProgressImage />
  </Flex>
));
ProgressPanelContent.displayName = 'ProgressPanelContent';

const mainPanelComponents: IDockviewReactProps['components'] = {
  welcome: InitialState,
  viewer: ViewerPanelContent,
  progress: ProgressPanelContent,
};

const onReadyMainPanel: IDockviewReactProps['onReady'] = (event) => {
  const { api } = event;
  api.addPanel({
    id: 'welcome',
    component: 'welcome',
    title: 'Launchpad',
  });
  api.addPanel({
    id: 'viewer',
    component: 'viewer',
    title: 'Image Viewer',
    position: {
      direction: 'within',
      referencePanel: 'welcome',
    },
  });
  api.addPanel({
    id: 'progress',
    component: 'progress',
    title: 'Generation Progress',
    position: {
      direction: 'within',
      referencePanel: 'welcome',
    },
  });

  const disposables = [
    api.onWillShowOverlay((e) => {
      if (e.kind === 'header_space' || e.kind === 'tab') {
        return;
      }
      e.preventDefault();
    }),
  ];

  return () => {
    disposables.forEach((disposable) => {
      disposable.dispose();
    });
  };
};

const MainPanel = memo(() => {
  return (
    <Flex w="full" h="full">
      <DockviewReact
        disableDnd={true}
        locked={true}
        disableFloatingGroups={true}
        dndEdges={false}
        defaultTabComponent={TabWithoutCloseButton}
        components={mainPanelComponents}
        onReady={onReadyMainPanel}
        theme={dockviewTheme}
      />
    </Flex>
  );
});
MainPanel.displayName = 'MainPanel';

const Left = memo(() => {
  return (
    <Flex flexDir="column" w="full" h="full" gap={2} py={2} pe={2}>
      <QueueControls />
      <Box position="relative" w="full" h="full">
        <ParametersPanelTextToImage />
      </Box>
    </Flex>
  );
});
Left.displayName = 'Left';

export const generateTabComponents: IGridviewReactProps['components'] = {
  left: Left,
  main: MainPanel,
  boards: BoardsListPanelContent,
  gallery: Gallery,
};

export const initializeGenerateTabLayout = (api: GridviewApi) => {
  const main = api.addPanel({
    id: 'main',
    component: 'main',
    minimumWidth: 256,
  });
  const left = api.addPanel({
    id: 'left',
    component: 'left',
    minimumWidth: LEFT_PANEL_MIN_SIZE_PX,
    position: {
      direction: 'left',
      referencePanel: 'main',
    },
  });
  api.addPanel({
    id: 'gallery',
    component: 'gallery',
    minimumWidth: RIGHT_PANEL_MIN_SIZE_PX,
    minimumHeight: 232,
    position: {
      direction: 'right',
      referencePanel: 'main',
    },
  });
  const boards = api.addPanel({
    id: 'boards',
    component: 'boards',
    minimumHeight: 36,
    position: {
      direction: 'above',
      referencePanel: 'gallery',
    },
  });
  left.api.setSize({ width: LEFT_PANEL_MIN_SIZE_PX });
  boards.api.setSize({ height: 256, width: RIGHT_PANEL_MIN_SIZE_PX });
};

export const GenerateTabAutoLayout = memo(() => {
  const [api, setApi] = useState<GridviewApi | null>(null);
  const onReady = useCallback<IGridviewReactProps['onReady']>((event) => {
    console.log('GenerateTabAutoLayout onReady');
    setApi(event.api);
    initializeGenerateTabLayout(event.api);
  }, []);
  return (
    <AutoLayoutProvider api={api}>
      <GridviewReact
        className="dockview-theme-invoke"
        components={generateTabComponents}
        onReady={onReady}
        orientation={Orientation.VERTICAL}
      />
    </AutoLayoutProvider>
  );
});
GenerateTabAutoLayout.displayName = 'GenerateTabAutoLayout';
