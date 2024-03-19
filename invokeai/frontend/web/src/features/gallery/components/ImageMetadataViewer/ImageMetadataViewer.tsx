import { ExternalLink, Flex, Tab, TabList, TabPanel, TabPanels, Tabs, Text } from '@invoke-ai/ui-library';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import { useMetadataItem } from 'features/metadata/hooks/useMetadataItem';
import { handlers } from 'features/metadata/util/handlers';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { useDebouncedMetadata } from 'services/api/hooks/useDebouncedMetadata';
import type { ImageDTO } from 'services/api/types';

import DataViewer from './DataViewer';
import ImageMetadataActions from './ImageMetadataActions';
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

  const { metadata } = useDebouncedMetadata(image.image_name);
  const createdBy = useMetadataItem(metadata, handlers.createdBy);

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
      {createdBy.valueOrNull && (
        <Text>
          {t('metadata.createdBy')}: {createdBy.valueOrNull}
        </Text>
      )}

      <Tabs variant="line" isLazy={true} display="flex" flexDir="column" w="full" h="full">
        <TabList>
          <Tab>{t('metadata.recallParameters')}</Tab>
          <Tab>{t('metadata.metadata')}</Tab>
          <Tab>{t('metadata.imageDetails')}</Tab>
          <Tab>{t('metadata.workflow')}</Tab>
        </TabList>

        <TabPanels>
          <TabPanel>
            {metadata ? (
              <ScrollableContent>
                <ImageMetadataActions metadata={metadata} />
              </ScrollableContent>
            ) : (
              <IAINoContentFallback label={t('metadata.noRecallParameters')} />
            )}
          </TabPanel>
          <TabPanel>
            {metadata ? (
              <DataViewer data={metadata} label={t('metadata.metadata')} />
            ) : (
              <IAINoContentFallback label={t('metadata.noMetaData')} />
            )}
          </TabPanel>
          <TabPanel>
            {image ? (
              <DataViewer data={image} label={t('metadata.imageDetails')} />
            ) : (
              <IAINoContentFallback label={t('metadata.noImageDetails')} />
            )}
          </TabPanel>
          <TabPanel>
            <ImageMetadataWorkflowTabContent image={image} />
          </TabPanel>
        </TabPanels>
      </Tabs>
    </Flex>
  );
};

export default memo(ImageMetadataViewer);
