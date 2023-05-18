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
import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { setIsLightboxOpen } from 'features/lightbox/store/lightboxSlice';
import { InvokeTabName } from 'features/ui/store/tabMap';
import { setActiveTab, togglePanels } from 'features/ui/store/uiSlice';
import { memo, ReactNode, useCallback, useMemo } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { MdDeviceHub, MdGridOn } from 'react-icons/md';
import { GoTextSize } from 'react-icons/go';
import {
  activeTabIndexSelector,
  activeTabNameSelector,
} from '../store/uiSelectors';
import { useTranslation } from 'react-i18next';
import { ResourceKey } from 'i18next';
import { requestCanvasRescale } from 'features/canvas/store/thunks/requestCanvasScale';
import { createSelector } from '@reduxjs/toolkit';
import { configSelector } from 'features/system/store/configSelectors';
import { isEqual } from 'lodash-es';
import { Panel, PanelGroup } from 'react-resizable-panels';
import ImageGalleryContent from 'features/gallery/components/ImageGalleryContent';
import TextToImageTab from './tabs/TextToImage/TextToImageTab';
import UnifiedCanvasTab from './tabs/UnifiedCanvas/UnifiedCanvasTab';
import NodesTab from './tabs/Nodes/NodesTab';
import { FaImage } from 'react-icons/fa';
import ResizeHandle from './tabs/ResizeHandle';
import ImageTab from './tabs/ImageToImage/ImageToImageTab';
import AuxiliaryProgressIndicator from 'app/components/AuxiliaryProgressIndicator';

export interface InvokeTabInfo {
  id: InvokeTabName;
  icon: ReactNode;
  content: ReactNode;
}

const tabs: InvokeTabInfo[] = [
  {
    id: 'txt2img',
    icon: <Icon as={GoTextSize} sx={{ boxSize: 6 }} />,
    content: <TextToImageTab />,
  },
  {
    id: 'img2img',
    icon: <Icon as={FaImage} sx={{ boxSize: 6 }} />,
    content: <ImageTab />,
  },
  {
    id: 'unifiedCanvas',
    icon: <Icon as={MdGridOn} sx={{ boxSize: 6 }} />,
    content: <UnifiedCanvasTab />,
  },
  {
    id: 'nodes',
    icon: <Icon as={MdDeviceHub} sx={{ boxSize: 6 }} />,
    content: <NodesTab />,
  },
];

const enabledTabsSelector = createSelector(
  configSelector,
  (config) => {
    const { disabledTabs } = config;

    return tabs.filter((tab) => !disabledTabs.includes(tab.id));
  },
  {
    memoizeOptions: { resultEqualityCheck: isEqual },
  }
);

const InvokeTabs = () => {
  const activeTab = useAppSelector(activeTabIndexSelector);
  const activeTabName = useAppSelector(activeTabNameSelector);
  const enabledTabs = useAppSelector(enabledTabsSelector);
  const isLightBoxOpen = useAppSelector(
    (state: RootState) => state.lightbox.isLightboxOpen
  );

  const { shouldPinGallery, shouldPinParametersPanel, shouldShowGallery } =
    useAppSelector((state: RootState) => state.ui);

  const { t } = useTranslation();

  const dispatch = useAppDispatch();

  // Lightbox Hotkey
  useHotkeys(
    'z',
    () => {
      dispatch(setIsLightboxOpen(!isLightBoxOpen));
    },
    [isLightBoxOpen]
  );

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

  const tabs = useMemo(
    () =>
      enabledTabs.map((tab) => (
        <Tooltip
          key={tab.id}
          hasArrow
          label={String(t(`common.${tab.id}` as ResourceKey))}
          placement="end"
        >
          <Tab>
            <VisuallyHidden>
              {String(t(`common.${tab.id}` as ResourceKey))}
            </VisuallyHidden>
            {tab.icon}
          </Tab>
        </Tooltip>
      )),
    [t, enabledTabs]
  );

  const tabPanels = useMemo(
    () =>
      enabledTabs.map((tab) => <TabPanel key={tab.id}>{tab.content}</TabPanel>),
    [enabledTabs]
  );

  return (
    <Tabs
      defaultIndex={activeTab}
      index={activeTab}
      onChange={(index: number) => {
        dispatch(setActiveTab(index));
      }}
      flexGrow={1}
      flexDir={{ base: 'column', xl: 'row' }}
      gap={{ base: 4 }}
      isLazy
    >
      <TabList
        pt={2}
        gap={4}
        flexDir={{ base: 'row', xl: 'column' }}
        justifyContent={{ base: 'center', xl: 'start' }}
      >
        {tabs}
        <Spacer />
        <AuxiliaryProgressIndicator />
      </TabList>
      <PanelGroup
        autoSaveId="app"
        direction="horizontal"
        style={{ height: '100%', width: '100%' }}
      >
        <Panel id="main">
          <TabPanels style={{ height: '100%', width: '100%' }}>
            {tabPanels}
          </TabPanels>
        </Panel>
        {shouldPinGallery && shouldShowGallery && (
          <>
            <ResizeHandle />
            <Panel
              onResize={handleResizeGallery}
              id="gallery"
              order={3}
              defaultSize={10}
              minSize={10}
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
