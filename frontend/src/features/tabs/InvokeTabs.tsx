import { Tab, TabPanel, TabPanels, Tabs } from '@chakra-ui/react';
import React, { ReactElement } from 'react';
import { ImageToImageWIP } from '../../common/components/WorkInProgress/ImageToImageWIP';
import InpaintingWIP from '../../common/components/WorkInProgress/InpaintingWIP';
import NodesWIP from '../../common/components/WorkInProgress/NodesWIP';
import OutpaintingWIP from '../../common/components/WorkInProgress/OutpaintingWIP';
import { PostProcessingWIP } from '../../common/components/WorkInProgress/PostProcessingWIP';
import TextToImage from './TextToImage/TextToImage';

export default function InvokeTabs() {
  const tab_dict = {
    txt2img: { title: 'Text To Image', panel: <TextToImage /> },
    img2img: { title: 'Image To Image', panel: <ImageToImageWIP /> },
    inpainting: { title: 'Inpainting', panel: <InpaintingWIP /> },
    outpainting: { title: 'Outpainting', panel: <OutpaintingWIP /> },
    nodes: { title: 'Nodes', panel: <NodesWIP /> },
    postprocess: { title: 'Post Processing', panel: <PostProcessingWIP /> },
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
    <Tabs className="app-tabs">
      <div className="app-tabs-list">{renderTabs()}</div>
      <TabPanels className="app-tabs-panels">{renderTabPanels()}</TabPanels>
    </Tabs>
  );
}
