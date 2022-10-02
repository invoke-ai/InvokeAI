import { Tab, TabPanel, TabPanels, Tabs } from '@chakra-ui/react';
import React, { ReactElement } from 'react';
import { ImageToImageWIP } from '../../common/components/WorkInProgress/ImageToImageWIP';
import InpaintingWIP from '../../common/components/WorkInProgress/InpaintingWIP';
import NodesWIP from '../../common/components/WorkInProgress/NodesWIP';
import OutpaintingWIP from '../../common/components/WorkInProgress/OutpaintingWIP';
import { PostProcessingWIP } from '../../common/components/WorkInProgress/PostProcessingWIP';
import ImageToImageIcon from '../../common/icons/ImageToImageIcon';
import InpaintIcon from '../../common/icons/InpaintIcon';
import NodesIcon from '../../common/icons/NodesIcon';
import OutpaintIcon from '../../common/icons/OutpaintIcon';
import PostprocessingIcon from '../../common/icons/PostprocessingIcon';
import TextToImageIcon from '../../common/icons/TextToImageIcon';
import TextToImage from './TextToImage/TextToImage';


export default function InvokeTabs() {
  const tab_dict = {
    txt2img: {
      title: <TextToImageIcon fill={'black'} boxSize={'2.5rem'} />,
      panel: <TextToImage />,
    },
    img2img: {
      title: <ImageToImageIcon fill={'black'} boxSize={'2.5rem'} />,
      panel: <ImageToImageWIP />,
    },
    inpainting: {
      title: <InpaintIcon fill={'black'} boxSize={'2.5rem'} />,
      panel: <InpaintingWIP />,
    },
    outpainting: {
      title: <OutpaintIcon fill={'black'} boxSize={'2.5rem'} />,
      panel: <OutpaintingWIP />,
    },
    nodes: {
      title: <NodesIcon fill={'black'} boxSize={'2.5rem'} />,
      panel: <NodesWIP />,
    },
    postprocess: {
      title: <PostprocessingIcon fill={'black'} boxSize={'2.5rem'} />,
      panel: <PostProcessingWIP />,
    },
  };

  const renderTabs = () => {
    const tabsToRender: ReactElement[] = [];
    Object.keys(tab_dict).forEach((key) => {
      tabsToRender.push(
        <Tab key={key}>{tab_dict[key as keyof typeof tab_dict].title}</Tab>
      );
    });
    return tabsToRender;
  };

  const renderTabPanels = () => {
    const tabPanelsToRender: ReactElement[] = [];
    Object.keys(tab_dict).forEach((key) => {
      tabPanelsToRender.push(
        <TabPanel className="app-tabs-panel" key={key}>
          {tab_dict[key as keyof typeof tab_dict].panel}
        </TabPanel>
      );
    });
    return tabPanelsToRender;
  };

  return (
    <Tabs className="app-tabs" variant={'unstyled'}>
      <div className="app-tabs-list">{renderTabs()}</div>
      <TabPanels className="app-tabs-panels">{renderTabPanels()}</TabPanels>
    </Tabs>
  );
}
