import { ExternalLink, Flex, Tab, TabList, TabPanel, TabPanels, Tabs } from '@invoke-ai/ui-library';
import { IAINoContentFallback, IAINoContentFallbackWithSpinner } from 'common/components/IAIImageFallback';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import ImageMetadataGraphTabContent from 'features/gallery/components/ImageMetadataViewer/ImageMetadataGraphTabContent';
import { MetadataHandlers } from 'features/metadata/parsing';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { useDebouncedMetadata } from 'services/api/hooks/useDebouncedMetadata';
import type { ImageDTO } from 'services/api/types';

import DataViewer from './DataViewer';
import ImageMetadataActions, { UnrecallableMetadataDatum } from './ImageMetadataActions';
import ImageMetadataWorkflowTabContent from './ImageMetadataWorkflowTabContent';

type ImageMetadataViewerProps = {
  image: ImageDTO;
};

const ImageMetadataViewer = ({ image }: ImageMetadataViewerProps) => {
  // TODO: fix hotkeys
  // const dispatch = useAppDispatch();
  // useHotkeys('esc', () => {
  //   dispatch(setShouldShowImageDetails(false));
  // });
  const { t } = useTranslation();

  const { metadata, isLoading } = useDebouncedMetadata(image.image_name);

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
      <ExternalLink href={image.image_url} label={image.image_name} />
      <UnrecallableMetadataDatum metadata={metadata} handler={MetadataHandlers.CreatedBy} />

      <Tabs variant="line" isLazy={true} display="flex" flexDir="column" w="full" h="full">
        <TabList>
          <Tab>{t('metadata.recallParameters')}</Tab>
          <Tab>{t('metadata.metadata')}</Tab>
          <Tab>{t('metadata.imageDetails')}</Tab>
          <Tab>{t('metadata.workflow')}</Tab>
          <Tab>{t('nodes.graph')}</Tab>
        </TabList>

        <TabPanels>
          <TabPanel>
            {isLoading && <IAINoContentFallbackWithSpinner label="Loading metadata..." />}
            {metadata && !isLoading && (
              <ScrollableContent>
                <ImageMetadataActions metadata={metadata} />
              </ScrollableContent>
            )}
            {!metadata && !isLoading && <IAINoContentFallback label={t('metadata.noRecallParameters')} />}
          </TabPanel>
          <TabPanel>
            {metadata ? (
              <DataViewer
                fileName={`${image.image_name.replace('.png', '')}_metadata`}
                data={metadata}
                label={t('metadata.metadata')}
              />
            ) : (
              <IAINoContentFallback label={t('metadata.noMetaData')} />
            )}
          </TabPanel>
          <TabPanel>
            {image ? (
              <DataViewer
                fileName={`${image.image_name.replace('.png', '')}_details`}
                data={image}
                label={t('metadata.imageDetails')}
              />
            ) : (
              <IAINoContentFallback label={t('metadata.noImageDetails')} />
            )}
          </TabPanel>
          <TabPanel>
            <ImageMetadataWorkflowTabContent image={image} />
          </TabPanel>
          <TabPanel>
            <ImageMetadataGraphTabContent image={image} />
          </TabPanel>
        </TabPanels>
      </Tabs>
    </Flex>
  );
};

export default memo(ImageMetadataViewer);
