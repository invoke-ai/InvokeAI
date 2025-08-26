import { ExternalLink, Flex, Tab, TabList, TabPanel, TabPanels, Tabs } from '@invoke-ai/ui-library';
import { IAINoContentFallback, IAINoContentFallbackWithSpinner } from 'common/components/IAIImageFallback';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import { ImageMetadataHandlers } from 'features/metadata/parsing';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { useDebouncedVideoMetadata } from 'services/api/hooks/useDebouncedMetadata';
import type { VideoDTO } from 'services/api/types';

import DataViewer from './DataViewer';
import { UnrecallableMetadataDatum, VideoMetadataActions } from './ImageMetadataActions';

type VideoMetadataViewerProps = {
  video: VideoDTO;
};

const VideoMetadataViewer = ({ video }: VideoMetadataViewerProps) => {
  const { t } = useTranslation();

  const { metadata, isLoading } = useDebouncedVideoMetadata(video.video_id);

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
      <ExternalLink href={video.video_url} label={video.video_id} />
      <UnrecallableMetadataDatum metadata={metadata} handler={ImageMetadataHandlers.CreatedBy} />

      <Tabs variant="line" isLazy={true} display="flex" flexDir="column" w="full" h="full">
        <TabList>
          <Tab>{t('metadata.recallParameters')}</Tab>
          <Tab>{t('metadata.metadata')}</Tab>
          <Tab>{t('metadata.imageDetails')}</Tab>
        </TabList>

        <TabPanels>
          <TabPanel>
            {isLoading && <IAINoContentFallbackWithSpinner label="Loading metadata..." />}
            {metadata && !isLoading && (
              <ScrollableContent>
                <VideoMetadataActions metadata={metadata} />
              </ScrollableContent>
            )}
            {!metadata && !isLoading && <IAINoContentFallback label={t('metadata.noRecallParameters')} />}
          </TabPanel>
          <TabPanel>
            {metadata ? (
              <DataViewer
                fileName={`${video.video_id.replace('.png', '')}_metadata`}
                data={metadata}
                label={t('metadata.metadata')}
              />
            ) : (
              <IAINoContentFallback label={t('metadata.noMetaData')} />
            )}
          </TabPanel>
          <TabPanel>
            {video ? (
              <DataViewer
                fileName={`${video.video_id.replace('.png', '')}_details`}
                data={video}
                label={t('metadata.imageDetails')}
              />
            ) : (
              <IAINoContentFallback label={t('metadata.noImageDetails')} />
            )}
          </TabPanel>
          {/* <TabPanel>
            <ImageMetadataWorkflowTabContent image={image} />
          </TabPanel>
          <TabPanel>
            <ImageMetadataGraphTabContent image={image} />
          </TabPanel> */}
        </TabPanels>
      </Tabs>
    </Flex>
  );
};

export default memo(VideoMetadataViewer);
