import {
  Flex,
  Tab,
  TabList,
  TabPanel,
  TabPanels,
  Tabs,
} from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import ImageMetadataJSON from 'features/gallery/components/ImageMetadataViewer/ImageMetadataJSON';
import { memo } from 'react';

const selector = createSelector(
  stateSelector,
  ({ nodes }) => {
    const lastSelectedNodeId =
      nodes.selectedNodes[nodes.selectedNodes.length - 1];

    const lastSelectedNode = nodes.nodes.find(
      (node) => node.id === lastSelectedNodeId
    );

    const lastSelectedNodeTemplate = lastSelectedNode
      ? nodes.nodeTemplates[lastSelectedNode.data.type]
      : undefined;

    return {
      node: lastSelectedNode,
      template: lastSelectedNodeTemplate,
    };
  },
  defaultSelectorOptions
);

const InspectorPanel = () => {
  const { node, template } = useAppSelector(selector);

  return (
    <Flex
      layerStyle="first"
      sx={{
        w: 'full',
        h: 'full',
        borderRadius: 'base',
        p: 4,
      }}
    >
      <Tabs
        variant="line"
        sx={{ display: 'flex', flexDir: 'column', w: 'full', h: 'full' }}
      >
        <TabList>
          <Tab>Node Template</Tab>
          <Tab>Node Data</Tab>
        </TabList>

        <TabPanels>
          <TabPanel>
            {template ? (
              <Flex
                sx={{
                  flexDir: 'column',
                  alignItems: 'flex-start',
                  gap: 2,
                  h: 'full',
                }}
              >
                <ImageMetadataJSON
                  jsonObject={template}
                  label="Node Template"
                />
              </Flex>
            ) : (
              <IAINoContentFallback
                label={
                  node
                    ? 'No template found for selected node'
                    : 'No node selected'
                }
                icon={null}
              />
            )}
          </TabPanel>
          <TabPanel>
            {node ? (
              <ImageMetadataJSON jsonObject={node.data} label="Node Data" />
            ) : (
              <IAINoContentFallback label="No node selected" icon={null} />
            )}
          </TabPanel>
        </TabPanels>
      </Tabs>
    </Flex>
  );
};

export default memo(InspectorPanel);
