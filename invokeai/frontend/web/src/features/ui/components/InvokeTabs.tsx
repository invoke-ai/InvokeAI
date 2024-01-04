import { Flex, Spacer } from '@chakra-ui/react';
import { useStore } from '@nanostores/react';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { $customNavComponent } from 'app/store/nanostores/customNavComponent';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InvIconButton } from 'common/components/InvIconButton/InvIconButton';
import { InvTab } from 'common/components/InvTabs/InvTab';
import {
  InvTabList,
  InvTabPanel,
  InvTabPanels,
  InvTabs,
} from 'common/components/InvTabs/wrapper';
import { InvTooltip } from 'common/components/InvTooltip/InvTooltip';
import ImageGalleryContent from 'features/gallery/components/ImageGalleryContent';
import NodeEditorPanelGroup from 'features/nodes/components/sidePanel/NodeEditorPanelGroup';
import InvokeAILogoComponent from 'features/system/components/InvokeAILogoComponent';
import SettingsMenu from 'features/system/components/SettingsModal/SettingsMenu';
import StatusIndicator from 'features/system/components/StatusIndicator';
import FloatingGalleryButton from 'features/ui/components/FloatingGalleryButton';
import FloatingParametersPanelButtons from 'features/ui/components/FloatingParametersPanelButtons';
import type { UsePanelOptions } from 'features/ui/hooks/usePanel';
import { usePanel } from 'features/ui/hooks/usePanel';
import { usePanelStorage } from 'features/ui/hooks/usePanelStorage';
import type { InvokeTabName } from 'features/ui/store/tabMap';
import {
  activeTabIndexSelector,
  activeTabNameSelector,
} from 'features/ui/store/uiSelectors';
import { setActiveTab } from 'features/ui/store/uiSlice';
import type { CSSProperties, MouseEvent, ReactElement, ReactNode } from 'react';
import { memo, useCallback, useMemo, useRef } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { FaCube, FaFont, FaImage } from 'react-icons/fa';
import { FaCircleNodes, FaList } from 'react-icons/fa6';
import { MdGridOn } from 'react-icons/md';
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
}

const tabs: InvokeTabInfo[] = [
  {
    id: 'txt2img',
    translationKey: 'common.txt2img',
    icon: <FaFont />,
    content: <TextToImageTab />,
  },
  {
    id: 'img2img',
    translationKey: 'common.img2img',
    icon: <FaImage />,
    content: <ImageTab />,
  },
  {
    id: 'unifiedCanvas',
    translationKey: 'common.unifiedCanvas',
    icon: <MdGridOn />,
    content: <UnifiedCanvasTab />,
  },
  {
    id: 'nodes',
    translationKey: 'common.nodes',
    icon: <FaCircleNodes />,
    content: <NodesTab />,
  },
  {
    id: 'modelManager',
    translationKey: 'modelManager.modelManager',
    icon: <FaCube />,
    content: <ModelManagerTab />,
  },
  {
    id: 'queue',
    translationKey: 'queue.queue',
    icon: <FaList />,
    content: <QueueTab />,
  },
];

const enabledTabsSelector = createMemoizedSelector(
  [stateSelector],
  ({ config }) => {
    const { disabledTabs } = config;
    const enabledTabs = tabs.filter((tab) => !disabledTabs.includes(tab.id));
    return enabledTabs;
  }
);

export const NO_GALLERY_PANEL_TABS: InvokeTabName[] = ['modelManager', 'queue'];
export const NO_OPTIONS_PANEL_TABS: InvokeTabName[] = ['modelManager', 'queue'];
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
  const shouldShowOptionsPanel = useMemo(
    () => !NO_OPTIONS_PANEL_TABS.includes(activeTabName),
    [activeTabName]
  );
  const shouldShowGalleryPanel = useMemo(
    () => !NO_GALLERY_PANEL_TABS.includes(activeTabName),
    [activeTabName]
  );

  const tabs = useMemo(
    () =>
      enabledTabs.map((tab) => (
        <InvTooltip key={tab.id} label={t(tab.translationKey)} placement="end">
          <InvTab
            as={InvIconButton}
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
        </InvTooltip>
      )),
    [enabledTabs, t, handleClickTab, activeTabName]
  );

  const tabPanels = useMemo(
    () =>
      enabledTabs.map((tab) => (
        <InvTabPanel key={tab.id}>{tab.content}</InvTabPanel>
      )),
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
    <InvTabs
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
        <InvTabList gap={4} pt={6} h="full" flexDir="column">
          {tabs}
        </InvTabList>
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
        {shouldShowOptionsPanel && (
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
              {activeTabName === 'nodes' ? (
                <NodeEditorPanelGroup />
              ) : (
                <ParametersPanel />
              )}
            </Panel>
            <ResizeHandle
              id="options-main-handle"
              onDoubleClick={optionsPanel.onDoubleClickHandle}
              orientation="vertical"
            />
          </>
        )}
        <Panel id="main-panel" order={1} minSize={20}>
          <InvTabPanels w="full" h="full">
            {tabPanels}
          </InvTabPanels>
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
      {shouldShowOptionsPanel && (
        <FloatingParametersPanelButtons panelApi={optionsPanel} />
      )}
      {shouldShowGalleryPanel && (
        <FloatingGalleryButton panelApi={galleryPanel} />
      )}
    </InvTabs>
  );
};

export default memo(InvokeTabs);
