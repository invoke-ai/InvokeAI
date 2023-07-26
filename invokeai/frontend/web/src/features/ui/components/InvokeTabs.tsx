import {
  Icon,
  Spacer,
  Tab,
  TabList,
  TabPanel,
  TabPanels,
  Tabs,
  Tooltip,
  VisuallyHidden,
} from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import AuxiliaryProgressIndicator from 'app/components/AuxiliaryProgressIndicator';
import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { requestCanvasRescale } from 'features/canvas/store/thunks/requestCanvasScale';
import ImageGalleryContent from 'features/gallery/components/ImageGalleryContent';
import { configSelector } from 'features/system/store/configSelectors';
import { InvokeTabName, tabMap } from 'features/ui/store/tabMap';
import { setActiveTab, togglePanels } from 'features/ui/store/uiSlice';
import { ResourceKey } from 'i18next';
import { isEqual } from 'lodash-es';
import { MouseEvent, ReactNode, memo, useCallback, useMemo } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { FaCube, FaFont, FaImage } from 'react-icons/fa';
import { MdDeviceHub, MdGridOn } from 'react-icons/md';
import { Panel, PanelGroup } from 'react-resizable-panels';
import { useMinimumPanelSize } from '../hooks/useMinimumPanelSize';
import {
  activeTabIndexSelector,
  activeTabNameSelector,
} from '../store/uiSelectors';
import ImageTab from './tabs/ImageToImage/ImageToImageTab';
import ModelManagerTab from './tabs/ModelManager/ModelManagerTab';
import NodesTab from './tabs/Nodes/NodesTab';
import ResizeHandle from './tabs/ResizeHandle';
import TextToImageTab from './tabs/TextToImage/TextToImageTab';
import UnifiedCanvasTab from './tabs/UnifiedCanvas/UnifiedCanvasTab';
import { systemSelector } from '../../system/store/systemSelectors';

export interface InvokeTabInfo {
  id: InvokeTabName;
  translationKey: string;
  icon: ReactNode;
  content: ReactNode;
}

const tabs: InvokeTabInfo[] = [
  {
    id: 'txt2img',
    translationKey: 'common.txt2img',
    icon: <Icon as={FaFont} sx={{ boxSize: 6, pointerEvents: 'none' }} />,
    content: <TextToImageTab />,
  },
  {
    id: 'img2img',
    translationKey: 'common.img2img',
    icon: <Icon as={FaImage} sx={{ boxSize: 6, pointerEvents: 'none' }} />,
    content: <ImageTab />,
  },
  {
    id: 'unifiedCanvas',
    translationKey: 'common.unifiedCanvas',
    icon: <Icon as={MdGridOn} sx={{ boxSize: 6, pointerEvents: 'none' }} />,
    content: <UnifiedCanvasTab />,
  },
  {
    id: 'nodes',
    translationKey: 'common.nodes',
    icon: <Icon as={MdDeviceHub} sx={{ boxSize: 6, pointerEvents: 'none' }} />,
    content: <NodesTab />,
  },
  {
    id: 'modelManager',
    translationKey: 'modelManager.modelManager',
    icon: <Icon as={FaCube} sx={{ boxSize: 6, pointerEvents: 'none' }} />,
    content: <ModelManagerTab />,
  },
  // {
  //   id: 'batch',
  //   icon: <Icon as={FaLayerGroup} sx={{ boxSize: 6, pointerEvents: 'none' }} />,
  //   content: <BatchTab />,
  // },
];

const enabledTabsSelector = createSelector(
  [configSelector, systemSelector],
  (config, system) => {
    const { disabledTabs } = config;
    const { isNodesEnabled } = system;

    const enabledTabs = tabs.filter((tab) => {
      if (tab.id === 'nodes') {
        return isNodesEnabled && !disabledTabs.includes(tab.id);
      } else {
        return !disabledTabs.includes(tab.id);
      }
    });

    return enabledTabs;
  },
  {
    memoizeOptions: { resultEqualityCheck: isEqual },
  }
);

const MIN_GALLERY_WIDTH = 350;
const DEFAULT_GALLERY_PCT = 20;
export const NO_GALLERY_TABS: InvokeTabName[] = ['modelManager'];

const InvokeTabs = () => {
  const activeTab = useAppSelector(activeTabIndexSelector);
  const activeTabName = useAppSelector(activeTabNameSelector);
  const enabledTabs = useAppSelector(enabledTabsSelector);

  const { shouldPinGallery, shouldPinParametersPanel, shouldShowGallery } =
    useAppSelector((state: RootState) => state.ui);

  const { t } = useTranslation();

  const dispatch = useAppDispatch();

  useHotkeys(
    'f',
    () => {
      dispatch(togglePanels());
      (shouldPinGallery || shouldPinParametersPanel) &&
        dispatch(requestCanvasRescale());
    },
    [shouldPinGallery, shouldPinParametersPanel]
  );

  const handleResizeGallery = useCallback(() => {
    if (activeTabName === 'unifiedCanvas') {
      dispatch(requestCanvasRescale());
    }
  }, [dispatch, activeTabName]);

  const handleClickTab = useCallback((e: MouseEvent<HTMLElement>) => {
    if (e.target instanceof HTMLElement) {
      e.target.blur();
    }
  }, []);

  const tabs = useMemo(
    () =>
      enabledTabs.map((tab) => (
        <Tooltip
          key={tab.id}
          hasArrow
          label={String(t(tab.translationKey as ResourceKey))}
          placement="end"
        >
          <Tab onClick={handleClickTab}>
            <VisuallyHidden>
              {String(t(tab.translationKey as ResourceKey))}
            </VisuallyHidden>
            {tab.icon}
          </Tab>
        </Tooltip>
      )),
    [enabledTabs, t, handleClickTab]
  );

  const tabPanels = useMemo(
    () =>
      enabledTabs.map((tab) => <TabPanel key={tab.id}>{tab.content}</TabPanel>),
    [enabledTabs]
  );

  const { ref: galleryPanelRef, minSizePct: galleryMinSizePct } =
    useMinimumPanelSize(MIN_GALLERY_WIDTH, DEFAULT_GALLERY_PCT, 'app');

  const handleTabChange = useCallback(
    (index: number) => {
      const activeTabName = tabMap[index];
      if (!activeTabName) {
        return;
      }
      dispatch(setActiveTab(activeTabName));
    },
    [dispatch]
  );

  return (
    <Tabs
      defaultIndex={activeTab}
      index={activeTab}
      onChange={handleTabChange}
      sx={{
        flexGrow: 1,
        gap: 4,
      }}
      isLazy
    >
      <TabList
        sx={{
          pt: 2,
          gap: 4,
          flexDir: 'column',
        }}
      >
        {tabs}
        <Spacer />
        <AuxiliaryProgressIndicator />
      </TabList>
      <PanelGroup
        id="app"
        autoSaveId="app"
        direction="horizontal"
        style={{ height: '100%', width: '100%' }}
      >
        <Panel id="main">
          <TabPanels style={{ height: '100%', width: '100%' }}>
            {tabPanels}
          </TabPanels>
        </Panel>
        {shouldPinGallery &&
          shouldShowGallery &&
          !NO_GALLERY_TABS.includes(activeTabName) && (
            <>
              <ResizeHandle />
              <Panel
                ref={galleryPanelRef}
                onResize={handleResizeGallery}
                id="gallery"
                order={3}
                defaultSize={
                  galleryMinSizePct > DEFAULT_GALLERY_PCT
                    ? galleryMinSizePct
                    : DEFAULT_GALLERY_PCT
                }
                minSize={galleryMinSizePct}
                maxSize={50}
              >
                <ImageGalleryContent />
              </Panel>
            </>
          )}
      </PanelGroup>
    </Tabs>
  );
};

export default memo(InvokeTabs);
