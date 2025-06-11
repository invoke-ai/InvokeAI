import { Box, ContextMenu, Divider, Flex, IconButton, Menu, MenuButton, MenuList } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import type { GridviewApi, IDockviewReactProps, IGridviewReactProps } from 'dockview';
import { DockviewReact, GridviewReact, Orientation } from 'dockview';
import { CanvasAlertsInvocationProgress } from 'features/controlLayers/components/CanvasAlerts/CanvasAlertsInvocationProgress';
import { CanvasAlertsPreserveMask } from 'features/controlLayers/components/CanvasAlerts/CanvasAlertsPreserveMask';
import { CanvasAlertsSelectedEntityStatus } from 'features/controlLayers/components/CanvasAlerts/CanvasAlertsSelectedEntityStatus';
import { CanvasContextMenuGlobalMenuItems } from 'features/controlLayers/components/CanvasContextMenu/CanvasContextMenuGlobalMenuItems';
import { CanvasContextMenuSelectedEntityMenuItems } from 'features/controlLayers/components/CanvasContextMenu/CanvasContextMenuSelectedEntityMenuItems';
import { CanvasDropArea } from 'features/controlLayers/components/CanvasDropArea';
import { CanvasLayersPanelContent } from 'features/controlLayers/components/CanvasLayersPanelContent';
import { Filter } from 'features/controlLayers/components/Filters/Filter';
import { CanvasHUD } from 'features/controlLayers/components/HUD/CanvasHUD';
import { InvokeCanvasComponent } from 'features/controlLayers/components/InvokeCanvasComponent';
import { SelectObject } from 'features/controlLayers/components/SelectObject/SelectObject';
import { CanvasSessionContextProvider } from 'features/controlLayers/components/SimpleSession/context';
import { InitialState } from 'features/controlLayers/components/SimpleSession/InitialState';
import { StagingAreaItemsList } from 'features/controlLayers/components/SimpleSession/StagingAreaItemsList';
import { StagingAreaToolbar } from 'features/controlLayers/components/StagingArea/StagingAreaToolbar';
import { CanvasToolbar } from 'features/controlLayers/components/Toolbar/CanvasToolbar';
import { Transform } from 'features/controlLayers/components/Transform/Transform';
import { CanvasManagerProviderGate } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { selectDynamicGrid, selectShowHUD } from 'features/controlLayers/store/canvasSettingsSlice';
import { selectCanvasSessionId } from 'features/controlLayers/store/canvasStagingAreaSlice';
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
import { PiDotsThreeOutlineVerticalFill } from 'react-icons/pi';

const MenuContent = memo(() => {
  return (
    <CanvasManagerProviderGate>
      <MenuList>
        <CanvasContextMenuSelectedEntityMenuItems />
        <CanvasContextMenuGlobalMenuItems />
      </MenuList>
    </CanvasManagerProviderGate>
  );
});
MenuContent.displayName = 'MenuContent';

const canvasBgSx = {
  position: 'relative',
  w: 'full',
  h: 'full',
  borderRadius: 'base',
  overflow: 'hidden',
  bg: 'base.900',
  '&[data-dynamic-grid="true"]': {
    bg: 'base.850',
  },
};

export const CanvasPanel = memo(() => {
  const dynamicGrid = useAppSelector(selectDynamicGrid);
  const showHUD = useAppSelector(selectShowHUD);
  const canvasId = useAppSelector(selectCanvasSessionId);

  const renderMenu = useCallback(() => {
    return <MenuContent />;
  }, []);

  return (
    <Flex
      tabIndex={-1}
      borderRadius="base"
      position="relative"
      flexDirection="column"
      height="full"
      width="full"
      gap={2}
      alignItems="center"
      justifyContent="center"
      overflow="hidden"
    >
      <CanvasManagerProviderGate>
        <CanvasToolbar />
      </CanvasManagerProviderGate>
      <Divider />
      <ContextMenu<HTMLDivElement> renderMenu={renderMenu} withLongPress={false}>
        {(ref) => (
          <Flex ref={ref} sx={canvasBgSx} data-dynamic-grid={dynamicGrid}>
            <InvokeCanvasComponent />
            <CanvasManagerProviderGate>
              <Flex
                position="absolute"
                flexDir="column"
                top={1}
                insetInlineStart={1}
                pointerEvents="none"
                gap={2}
                alignItems="flex-start"
              >
                {showHUD && <CanvasHUD />}
                <CanvasAlertsSelectedEntityStatus />
                <CanvasAlertsPreserveMask />
                <CanvasAlertsInvocationProgress />
              </Flex>
              <Flex position="absolute" top={1} insetInlineEnd={1}>
                <Menu>
                  <MenuButton as={IconButton} icon={<PiDotsThreeOutlineVerticalFill />} colorScheme="base" />
                  <MenuContent />
                </Menu>
              </Flex>
            </CanvasManagerProviderGate>
          </Flex>
        )}
      </ContextMenu>
      {canvasId !== null && (
        <CanvasManagerProviderGate>
          <CanvasSessionContextProvider type="advanced" id={canvasId}>
            <Flex
              position="absolute"
              flexDir="column"
              bottom={4}
              gap={2}
              align="center"
              justify="center"
              left={4}
              right={4}
            >
              <Flex position="relative" maxW="full" w="full" h={108}>
                <StagingAreaItemsList />
              </Flex>
              <Flex gap={2}>
                <StagingAreaToolbar />
              </Flex>
            </Flex>
          </CanvasSessionContextProvider>
        </CanvasManagerProviderGate>
      )}
      <Flex position="absolute" bottom={4}>
        <CanvasManagerProviderGate>
          <Filter />
          <Transform />
          <SelectObject />
        </CanvasManagerProviderGate>
      </Flex>
      <CanvasManagerProviderGate>
        <CanvasDropArea />
      </CanvasManagerProviderGate>
    </Flex>
  );
});
CanvasPanel.displayName = 'CanvasPanel';

const LayersPanelContent = memo(() => (
  <CanvasManagerProviderGate>
    <CanvasLayersPanelContent />
  </CanvasManagerProviderGate>
));
LayersPanelContent.displayName = 'LayersPanelContent';

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
  canvas: CanvasPanel,
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
    id: 'canvas',
    component: 'canvas',
    title: 'Canvas',
    position: {
      direction: 'within',
      referencePanel: 'welcome',
    },
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

export const canvasTabComponents: IGridviewReactProps['components'] = {
  left: Left,
  main: MainPanel,
  boards: BoardsListPanelContent,
  gallery: Gallery,
  layers: LayersPanelContent,
};

export const initializeCanvasTabLayout = (api: GridviewApi) => {
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
  api.addPanel({
    id: 'layers',
    component: 'layers',
    minimumHeight: 256,
    position: {
      direction: 'below',
      referencePanel: 'gallery',
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

export const CanvasTabAutoLayout = memo(() => {
  const [api, setApi] = useState<GridviewApi | null>(null);
  const onReady = useCallback<IGridviewReactProps['onReady']>((event) => {
    setApi(event.api);
    initializeCanvasTabLayout(event.api);
  }, []);
  return (
    <AutoLayoutProvider api={api}>
      <GridviewReact
        className="dockview-theme-invoke"
        components={canvasTabComponents}
        onReady={onReady}
        orientation={Orientation.VERTICAL}
      />
    </AutoLayoutProvider>
  );
});
CanvasTabAutoLayout.displayName = 'CanvasTabAutoLayout';
