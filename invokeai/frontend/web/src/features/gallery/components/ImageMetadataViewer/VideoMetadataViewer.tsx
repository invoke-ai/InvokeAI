import { ExternalLink, Flex, Tab, TabList, TabPanel, TabPanels, Tabs, Text } from '@invoke-ai/ui-library';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { useGetVideoMetadataQuery, useGetVideoWorkflowQuery } from 'services/api/endpoints/videos';
import type { VideoDTO } from 'services/api/types';

import DataViewer from './DataViewer';

type Props = {
  video: VideoDTO;
};

/**
 * Details overlay for a selected video. Counterpart to ImageMetadataViewer, trimmed to what
 * videos carry: the raw generation metadata and the saved workflow/graph. (Per-field recall
 * is an image-only concept for now.)
 */
const VideoMetadataViewer = ({ video }: Props) => {
  const { t } = useTranslation();
  const { data: metadata } = useGetVideoMetadataQuery(video.video_name);
  const { data: workflowAndGraph } = useGetVideoWorkflowQuery(video.video_name, { skip: !video.has_workflow });

  return (
    <Flex
      layerStyle="first"
      padding={4}
      gap={1}
      flexDirection="column"
      width="full"
      height="full"
      borderRadius="base"
      position="absolute"
      overflow="hidden"
    >
      <ExternalLink href={video.video_url} label={video.video_name} />

      <Tabs variant="line" sx={{ display: 'flex', flexDir: 'column', w: 'full', h: 'full' }}>
        <TabList>
          <Tab>{t('metadata.metadata')}</Tab>
          <Tab>{t('metadata.workflow')}</Tab>
          <Tab>{t('nodes.graph')}</Tab>
        </TabList>

        <TabPanels>
          <TabPanel>
            {metadata ? (
              <DataViewer
                fileName={`${video.video_name.replace('.mp4', '')}_metadata`}
                data={metadata}
                label={t('metadata.metadata')}
              />
            ) : (
              <IAINoContentFallback label={t('metadata.noMetaData')} />
            )}
          </TabPanel>
          <TabPanel>
            {workflowAndGraph?.workflow ? (
              <DataViewer
                fileName={`${video.video_name.replace('.mp4', '')}_workflow`}
                data={workflowAndGraph.workflow}
                label={t('metadata.workflow')}
              />
            ) : (
              <IAINoContentFallback label={t('nodes.noWorkflow')} />
            )}
          </TabPanel>
          <TabPanel>
            {workflowAndGraph?.graph ? (
              <DataViewer
                fileName={`${video.video_name.replace('.mp4', '')}_graph`}
                data={workflowAndGraph.graph}
                label={t('nodes.graph')}
              />
            ) : (
              <IAINoContentFallback label={t('nodes.noGraph')} />
            )}
          </TabPanel>
        </TabPanels>
      </Tabs>
      <Text fontSize="xs" color="base.400">
        {video.video_name}
      </Text>
    </Flex>
  );
};

export default memo(VideoMetadataViewer);
