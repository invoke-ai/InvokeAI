import {
  ChakraProps,
  Icon,
  Tab,
  TabList,
  TabPanel,
  TabPanels,
  Tabs,
  Tooltip,
  VisuallyHidden,
} from '@chakra-ui/react';
import { RootState } from 'app/store';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import NodesWIP from 'common/components/WorkInProgress/NodesWIP';
import { PostProcessingWIP } from 'common/components/WorkInProgress/PostProcessingWIP';
import TrainingWIP from 'common/components/WorkInProgress/Training';
import { setIsLightboxOpen } from 'features/lightbox/store/lightboxSlice';
import { InvokeTabName } from 'features/ui/store/tabMap';
import { setActiveTab, togglePanels } from 'features/ui/store/uiSlice';
import { ReactNode, useMemo } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import {
  MdDeviceHub,
  MdFlashOn,
  MdGridOn,
  MdPhotoFilter,
  MdPhotoLibrary,
  MdTextFields,
} from 'react-icons/md';
import { activeTabIndexSelector } from '../store/uiSelectors';
import ImageToImageWorkarea from 'features/ui/components/tabs/ImageToImage/ImageToImageWorkarea';
import TextToImageWorkarea from 'features/ui/components/tabs/TextToImage/TextToImageWorkarea';
import UnifiedCanvasWorkarea from 'features/ui/components/tabs/UnifiedCanvas/UnifiedCanvasWorkarea';
import { useTranslation } from 'react-i18next';
import { ResourceKey } from 'i18next';
import { requestCanvasRescale } from 'features/canvas/store/thunks/requestCanvasScale';

export interface InvokeTabInfo {
  id: InvokeTabName;
  icon: ReactNode;
  workarea: ReactNode;
}

const tabIconStyles: ChakraProps['sx'] = {
  boxSize: 6,
};

const tabInfo: InvokeTabInfo[] = [
  {
    id: 'txt2img',
    icon: <Icon as={MdTextFields} sx={tabIconStyles} />,
    workarea: <TextToImageWorkarea />,
  },
  {
    id: 'img2img',
    icon: <Icon as={MdPhotoLibrary} sx={tabIconStyles} />,
    workarea: <ImageToImageWorkarea />,
  },
  {
    id: 'unifiedCanvas',
    icon: <Icon as={MdGridOn} sx={tabIconStyles} />,
    workarea: <UnifiedCanvasWorkarea />,
  },
  {
    id: 'nodes',
    icon: <Icon as={MdDeviceHub} sx={tabIconStyles} />,
    workarea: <NodesWIP />,
  },
  {
    id: 'postprocessing',
    icon: <Icon as={MdPhotoFilter} sx={tabIconStyles} />,
    workarea: <PostProcessingWIP />,
  },
  {
    id: 'training',
    icon: <Icon as={MdFlashOn} sx={tabIconStyles} />,
    workarea: <TrainingWIP />,
  },
];

export default function InvokeTabs() {
  const activeTab = useAppSelector(activeTabIndexSelector);

  const isLightBoxOpen = useAppSelector(
    (state: RootState) => state.lightbox.isLightboxOpen
  );

  const shouldPinGallery = useAppSelector(
    (state: RootState) => state.ui.shouldPinGallery
  );

  const shouldPinParametersPanel = useAppSelector(
    (state: RootState) => state.ui.shouldPinParametersPanel
  );

  const { t } = useTranslation();

  const dispatch = useAppDispatch();

  useHotkeys('1', () => {
    dispatch(setActiveTab(0));
  });

  useHotkeys('2', () => {
    dispatch(setActiveTab(1));
  });

  useHotkeys('3', () => {
    dispatch(setActiveTab(2));
  });

  useHotkeys('4', () => {
    dispatch(setActiveTab(3));
  });

  useHotkeys('5', () => {
    dispatch(setActiveTab(4));
  });

  useHotkeys('6', () => {
    dispatch(setActiveTab(5));
  });

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

  const tabs = useMemo(
    () =>
      tabInfo.map((tab) => (
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
    [t]
  );

  const tabPanels = useMemo(
    () =>
      tabInfo.map((tab) => <TabPanel key={tab.id}>{tab.workarea}</TabPanel>),
    []
  );

  /**
   * isLazy means the tabs are mounted and unmounted when changing them. There is a tradeoff here,
   * as mounting is expensive, but so is retaining all tabs in the DOM at all times.
   *
   * Removing isLazy messes with the outside click watcher, which is used by ResizableDrawer.
   * Because you have multiple handlers listening for an outside click, any click anywhere triggers
   * the watcher for the hidden drawers, closing the open drawer.
   *
   * TODO: Add logic to the `useOutsideClick` in ResizableDrawer to enable it only for the active
   * tab's drawer.
   */
  return (
    <Tabs
      isLazy
      defaultIndex={activeTab}
      index={activeTab}
      onChange={(index: number) => {
        dispatch(setActiveTab(index));
      }}
      flexGrow={1}
    >
      <TabList>{tabs}</TabList>
      <TabPanels>{tabPanels}</TabPanels>
    </Tabs>
  );
}
