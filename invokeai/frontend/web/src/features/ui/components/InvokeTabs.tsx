import {
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
import Lightbox from 'features/lightbox/components/Lightbox';
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
  id: string;
  icon: ReactNode;
  workarea: ReactNode;
  tooltip: string;
}

const tabInfo: InvokeTabInfo[] = [
  {
    id: 'text2img',
    icon: <Icon as={MdTextFields} boxSize={6} />,
    workarea: <TextToImageWorkarea />,
    tooltip: 'Text To Image',
  },
  {
    id: 'img2img',
    icon: <Icon as={MdPhotoLibrary} boxSize={6} />,
    workarea: <ImageToImageWorkarea />,
    tooltip: 'Image To Image',
  },
  {
    id: 'unifiedCanvas',
    icon: <Icon as={MdGridOn} boxSize={6} />,
    workarea: <UnifiedCanvasWorkarea />,
    tooltip: 'Unified Canvas',
  },
  {
    id: 'nodes',
    icon: <Icon as={MdDeviceHub} boxSize={6} />,
    workarea: <NodesWIP />,
    tooltip: 'Nodes',
  },
  {
    id: 'postProcessing',
    icon: <Icon as={MdPhotoFilter} boxSize={6} />,
    workarea: <PostProcessingWIP />,
    tooltip: 'Post Processing',
  },
  {
    id: 'training',
    icon: <Icon as={MdFlashOn} boxSize={6} />,
    workarea: <TrainingWIP />,
    tooltip: 'Training',
  },
];

export interface InvokeTabInfo2 {
  icon: ReactNode;
  workarea: ReactNode;
  tooltip: string;
}

export const tabDict: Record<InvokeTabName, InvokeTabInfo2> = {
  txt2img: {
    icon: <Icon as={MdTextFields} boxSize={6} />,
    workarea: <TextToImageWorkarea />,
    tooltip: 'Text To Image',
  },
  img2img: {
    icon: <Icon as={MdPhotoLibrary} boxSize={6} />,
    workarea: <ImageToImageWorkarea />,
    tooltip: 'Image To Image',
  },
  unifiedCanvas: {
    icon: <Icon as={MdGridOn} boxSize={6} />,
    workarea: <UnifiedCanvasWorkarea />,
    tooltip: 'Unified Canvas',
  },
  nodes: {
    icon: <Icon as={MdDeviceHub} boxSize={6} />,
    workarea: <NodesWIP />,
    tooltip: 'Nodes',
  },
  postprocess: {
    icon: <Icon as={MdPhotoFilter} boxSize={6} />,
    workarea: <PostProcessingWIP />,
    tooltip: 'Post Processing',
  },
  training: {
    icon: <Icon as={MdFlashOn} boxSize={6} />,
    workarea: <TrainingWIP />,
    tooltip: 'Training',
  },
};

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
            <VisuallyHidden>{tab.tooltip}</VisuallyHidden>
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
      <TabPanels>{isLightBoxOpen ? <Lightbox /> : tabPanels}</TabPanels>
    </Tabs>
  );
}
