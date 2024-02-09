import { Flex, IconButton, Spacer, Tab, TabList, TabPanel, TabPanels, Tabs, Tooltip } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { $customNavComponent } from 'app/store/nanostores/customNavComponent';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import ImageGalleryContent from 'features/gallery/components/ImageGalleryContent';
import NodeEditorPanelGroup from 'features/nodes/components/sidePanel/NodeEditorPanelGroup';
import InvokeAILogoComponent from 'features/system/components/InvokeAILogoComponent';
import SettingsMenu from 'features/system/components/SettingsModal/SettingsMenu';
import StatusIndicator from 'features/system/components/StatusIndicator';
import { selectConfigSlice } from 'features/system/store/configSlice';
import FloatingGalleryButton from 'features/ui/components/FloatingGalleryButton';
import FloatingParametersPanelButtons from 'features/ui/components/FloatingParametersPanelButtons';
import type { UsePanelOptions } from 'features/ui/hooks/usePanel';
import { usePanel } from 'features/ui/hooks/usePanel';
import { usePanelStorage } from 'features/ui/hooks/usePanelStorage';
import type { InvokeTabName } from 'features/ui/store/tabMap';
import { activeTabIndexSelector, activeTabNameSelector } from 'features/ui/store/uiSelectors';
import { setActiveTab } from 'features/ui/store/uiSlice';
import type { CSSProperties, MouseEvent, ReactElement, ReactNode } from 'react';
import { memo, useCallback, useMemo, useRef } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { PiFlowArrowBold } from 'react-icons/pi';
import { RiBox2Line, RiBrushLine, RiImage2Line, RiInputMethodLine, RiPlayList2Fill } from 'react-icons/ri';
import type { ImperativePanelGroupHandle } from 'react-resizable-panels';
import { Panel, PanelGroup } from 'react-resizable-panels';

import ParametersPanel from './ParametersPanel';
import ImageTab from './tabs/ImageToImageTab';
import ModelManagerTab from './tabs/ModelManagerTab';
import NodesTab from './tabs/NodesTab';
import QueueTab from './tabs/QueueTab';
import ResizeHandle from './tabs/ResizeHandle';
import TextToImageTab from './tabs/TextToImageTab';
import UnifiedCanvasTab from './tabs/UnifiedCanvasTab';

export interface InvokeTabInfo {
  id: InvokeTabName;
  translationKey: string;
  icon: ReactElement;
  content: ReactNode;
  optionsPanel?: ReactNode;
}

const tabs: InvokeTabInfo[] = [
  {
    id: 'txt2img',
    translationKey: 'common.txt2img',
    icon: <RiInputMethodLine />,
    content: <TextToImageTab />,
    optionsPanel: <ParametersPanel />,
  },
  {
    id: 'img2img',
    translationKey: 'common.img2img',
    icon: <RiImage2Line />,
    content: <ImageTab />,
    optionsPanel: <ParametersPanel />,
  },
  {
    id: 'unifiedCanvas',
    translationKey: 'common.unifiedCanvas',
    icon: <RiBrushLine />,
    content: <UnifiedCanvasTab />,
    optionsPanel: <ParametersPanel />,
  },
  {
    id: 'nodes',
    translationKey: 'common.nodes',
    icon: <PiFlowArrowBold />,
    content: <NodesTab />,
    optionsPanel: <NodeEditorPanelGroup />,
  },
  {
    id: 'modelManager',
    translationKey: 'modelManager.modelManager',
    icon: <RiBox2Line />,
    content: <ModelManagerTab />,
  },
  {
    id: 'queue',
    translationKey: 'queue.queue',
    icon: <RiPlayList2Fill />,
    content: <QueueTab />,
  },
];

const enabledTabsSelector = createMemoizedSelector(selectConfigSlice, (config) =>
  tabs.filter((tab) => !config.disabledTabs.includes(tab.id))
);

export const NO_GALLERY_PANEL_TABS: InvokeTabName[] = ['modelManager', 'queue'];
const panelStyles: CSSProperties = { height: '100%', width: '100%' };
const GALLERY_MIN_SIZE_PX = 310;
const GALLERY_MIN_SIZE_PCT = 20;
const OPTIONS_PANEL_MIN_SIZE_PX = 430;
const OPTIONS_PANEL_MIN_SIZE_PCT = 20;

const appPanelGroupId = 'app-panel-group';

const InvokeTabs = () => {
  const activeTabIndex = useAppSelector(activeTabIndexSelector);
  const activeTabName = useAppSelector(activeTabNameSelector);
  const enabledTabs = useAppSelector(enabledTabsSelector);
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const customNavComponent = useStore($customNavComponent);
  const panelGroupRef = useRef<ImperativePanelGroupHandle>(null);
  const handleClickTab = useCallback((e: MouseEvent<HTMLElement>) => {
    if (e.target instanceof HTMLElement) {
      e.target.blur();
    }
  }, []);
  const shouldShowGalleryPanel = useMemo(() => !NO_GALLERY_PANEL_TABS.includes(activeTabName), [activeTabName]);

  const tabs = useMemo(
    () =>
      enabledTabs.map((tab) => (
        <Tooltip key={tab.id} label={t(tab.translationKey)} placement="end">
          <Tab
            as={IconButton}
            p={0}
            onClick={handleClickTab}
            icon={tab.icon}
            size="md"
            fontSize="24px"
            variant="appTab"
            data-selected={activeTabName === tab.id}
            aria-label={t(tab.translationKey)}
            data-testid={t(tab.translationKey)}
          />
        </Tooltip>
      )),
    [enabledTabs, t, handleClickTab, activeTabName]
  );

  const activeTabData = useMemo(() => {
    return enabledTabs.find((tab) => tab.id === activeTabName);
  }, [enabledTabs, activeTabName]);

  const tabPanels = useMemo(
    () => enabledTabs.map((tab) => <TabPanel key={tab.id}>{tab.content}</TabPanel>),
    [enabledTabs]
  );

  const handleTabChange = useCallback(
    (index: number) => {
      const tab = enabledTabs[index];
      if (!tab) {
        return;
      }
      dispatch(setActiveTab(tab.id));
    },
    [dispatch, enabledTabs]
  );

  const optionsPanelUsePanelOptions = useMemo<UsePanelOptions>(
    () => ({
      unit: 'pixels',
      minSize: OPTIONS_PANEL_MIN_SIZE_PX,
      fallbackMinSizePct: OPTIONS_PANEL_MIN_SIZE_PCT,
      panelGroupRef,
      panelGroupDirection: 'horizontal',
    }),
    []
  );

  const galleryPanelUsePanelOptions = useMemo<UsePanelOptions>(
    () => ({
      unit: 'pixels',
      minSize: GALLERY_MIN_SIZE_PX,
      fallbackMinSizePct: GALLERY_MIN_SIZE_PCT,
      panelGroupRef,
      panelGroupDirection: 'horizontal',
    }),
    []
  );

  const panelStorage = usePanelStorage();

  const optionsPanel = usePanel(optionsPanelUsePanelOptions);

  const galleryPanel = usePanel(galleryPanelUsePanelOptions);

  useHotkeys('g', galleryPanel.toggle, [galleryPanel.toggle]);
  useHotkeys(['t', 'o'], optionsPanel.toggle, [optionsPanel.toggle]);
  useHotkeys(
    'shift+r',
    () => {
      optionsPanel.reset();
      galleryPanel.reset();
    },
    [optionsPanel.reset, galleryPanel.reset]
  );
  useHotkeys(
    'f',
    () => {
      if (optionsPanel.isCollapsed || galleryPanel.isCollapsed) {
        optionsPanel.expand();
        galleryPanel.expand();
      } else {
        optionsPanel.collapse();
        galleryPanel.collapse();
      }
    },
    [
      optionsPanel.isCollapsed,
      galleryPanel.isCollapsed,
      optionsPanel.expand,
      galleryPanel.expand,
      optionsPanel.collapse,
      galleryPanel.collapse,
    ]
  );

  return (
    <Tabs
      id="invoke-app-tabs"
      variant="appTabs"
      defaultIndex={activeTabIndex}
      index={activeTabIndex}
      onChange={handleTabChange}
      w="full"
      h="full"
      gap={4}
      p={4}
      isLazy
    >
      <Flex flexDir="column" alignItems="center" pt={4} pb={2} gap={4}>
        <InvokeAILogoComponent />
        <TabList gap={4} pt={6} h="full" flexDir="column">
          {tabs}
        </TabList>
        <Spacer />
        <StatusIndicator />
        {customNavComponent ? customNavComponent : <SettingsMenu />}
      </Flex>
      <PanelGroup
        ref={panelGroupRef}
        id={appPanelGroupId}
        autoSaveId="app"
        direction="horizontal"
        style={panelStyles}
        storage={panelStorage}
      >
        {!!activeTabData?.optionsPanel && (
          <>
            <Panel
              id="options-panel"
              ref={optionsPanel.ref}
              order={0}
              defaultSize={optionsPanel.minSize}
              minSize={optionsPanel.minSize}
              onCollapse={optionsPanel.onCollapse}
              onExpand={optionsPanel.onExpand}
              collapsible
            >
              {activeTabData?.optionsPanel}
            </Panel>
            <ResizeHandle
              id="options-main-handle"
              onDoubleClick={optionsPanel.onDoubleClickHandle}
              orientation="vertical"
            />
          </>
        )}
        <Panel id="main-panel" order={1} minSize={20}>
          <TabPanels w="full" h="full">
            {tabPanels}
          </TabPanels>
        </Panel>
        {shouldShowGalleryPanel && (
          <>
            <ResizeHandle
              id="main-gallery-handle"
              orientation="vertical"
              onDoubleClick={galleryPanel.onDoubleClickHandle}
            />
            <Panel
              id="gallery-panel"
              ref={galleryPanel.ref}
              order={2}
              defaultSize={galleryPanel.minSize}
              minSize={galleryPanel.minSize}
              onCollapse={galleryPanel.onCollapse}
              onExpand={galleryPanel.onExpand}
              collapsible
            >
              <ImageGalleryContent />
            </Panel>
          </>
        )}
      </PanelGroup>
      {!!activeTabData?.optionsPanel && <FloatingParametersPanelButtons panelApi={optionsPanel} />}
      {shouldShowGalleryPanel && <FloatingGalleryButton panelApi={galleryPanel} />}
    </Tabs>
  );
};

export default memo(InvokeTabs);
