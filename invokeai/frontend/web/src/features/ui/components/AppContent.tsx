import 'dockview/dist/styles/dockview.css';
import './dockview_theme_invoke.css';

import { Flex } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppSelector } from 'app/store/storeHooks';
import type { DockviewApi } from 'dockview';
import { useDndMonitor } from 'features/dnd/useDndMonitor';
import { GridviewWrapper } from 'features/ui/components/GridviewWrapper';
import { VerticalNavBar } from 'features/ui/components/VerticalNavBar';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
import { atom } from 'nanostores';
import { memo } from 'react';

export const $panels = atom<{ api: DockviewApi; resetLayout: () => void } | null>(null);

export const $advancedLayout = atom<boolean>(false);
export const toggleAdvancedLayout = () => {
  $advancedLayout.set(!$advancedLayout.get());
};

export const AppContent = memo(() => {
  useDndMonitor();
  const tab = useAppSelector(selectActiveTab);
  const advancedLayout = useStore($advancedLayout);
  // useRegisteredHotkeys({
  //   id: 'toggleLeftPanel',
  //   category: 'app',
  //   callback: leftPanel.toggle,
  //   options: { enabled: withLeftPanel },
  //   dependencies: [leftPanel.toggle, withLeftPanel],
  // });

  // useRegisteredHotkeys({
  //   id: 'toggleRightPanel',
  //   category: 'app',
  //   callback: rightPanel.toggle,
  //   options: { enabled: withRightPanel },
  //   dependencies: [rightPanel.toggle, withRightPanel],
  // });

  // useRegisteredHotkeys({
  //   id: 'resetPanelLayout',
  //   category: 'app',
  //   callback: () => {
  //     leftPanel.reset();
  //     rightPanel.reset();
  //   },
  //   dependencies: [leftPanel.reset, rightPanel.reset],
  // });
  // useRegisteredHotkeys({
  //   id: 'togglePanels',
  //   category: 'app',
  //   callback: () => {
  //     if (leftPanel.isCollapsed || rightPanel.isCollapsed) {
  //       leftPanel.expand();
  //       rightPanel.expand();
  //     } else {
  //       leftPanel.collapse();
  //       rightPanel.collapse();
  //     }
  //   },
  //   dependencies: [
  //     leftPanel.isCollapsed,
  //     rightPanel.isCollapsed,
  //     leftPanel.expand,
  //     rightPanel.expand,
  //     leftPanel.collapse,
  //     rightPanel.collapse,
  //   ],
  // });

  return (
    <Flex id="invoke-app-tabs" w="full" h="full" overflow="hidden" position="relative">
      <VerticalNavBar />
      {/* <DockviewReact
        components={components}
        onReady={onReady}
        theme={theme}
        defaultTabComponent={MyCustomTab}
        rightHeaderActionsComponent={RightHeaderActions}
      /> */}
      <GridviewWrapper />
      {/* <PanelGroup
        ref={imperativePanelGroupRef}
        id="app-panel-group"
        autoSaveId="app-panel-group"
        direction="horizontal"
        style={panelStyles}
      >
        {withLeftPanel && (
          <>
            <Panel id="left-panel" order={0} collapsible style={panelStyles} {...leftPanel.panelProps}>
              <LeftPanelContent />
            </Panel>
            <VerticalResizeHandle id="left-main-handle" {...leftPanel.resizeHandleProps} />
          </>
        )}
        <Panel id="main-panel" order={1} minSize={20} style={panelStyles}>
          <MainPanelContent />
          {withLeftPanel && <FloatingLeftPanelButtons onToggle={leftPanel.toggle} />}
          {withRightPanel && <FloatingRightPanelButtons onToggle={rightPanel.toggle} />}
        </Panel>
        {withRightPanel && (
          <>
            <VerticalResizeHandle id="main-right-handle" {...rightPanel.resizeHandleProps} />
            <Panel id="right-panel" order={2} style={panelStyles} collapsible {...rightPanel.panelProps}>
              <RightPanelContent />
            </Panel>
          </>
        )}
      </PanelGroup> */}
    </Flex>
  );
});
AppContent.displayName = 'AppContent';
