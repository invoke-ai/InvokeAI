import { ExternalLinkIcon } from '@chakra-ui/icons';
import {
  Flex,
  Link,
  Tab,
  TabList,
  TabPanel,
  TabPanels,
  Tabs,
  Text,
} from '@chakra-ui/react';
import { skipToken } from '@reduxjs/toolkit/dist/query';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import ScrollableContent from 'features/nodes/components/sidePanel/ScrollableContent';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { useGetImageMetadataQuery } from 'services/api/endpoints/images';
import { useGetWorkflowQuery } from 'services/api/endpoints/workflows';
import { ImageDTO } from 'services/api/types';
import { useDebounce } from 'use-debounce';
import DataViewer from './DataViewer';
import ImageMetadataActions from './ImageMetadataActions';

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

  const [debouncedImageName] = useDebounce(image.image_name, 300);
  const [debouncedWorkflowId] = useDebounce(image.workflow_id, 300);

  const { data: metadata } = useGetImageMetadataQuery(
    debouncedImageName ?? skipToken
  );

  const { data: workflow } = useGetWorkflowQuery(
    debouncedWorkflowId ?? skipToken
  );

  return (
    <Flex
      layerStyle="first"
      sx={{
        padding: 4,
        gap: 1,
        flexDirection: 'column',
        width: 'full',
        height: 'full',
        borderRadius: 'base',
        position: 'absolute',
        overflow: 'hidden',
      }}
    >
      <Flex gap={2}>
        <Text fontWeight="semibold">File:</Text>
        <Link href={image.image_url} isExternal maxW="calc(100% - 3rem)">
          {image.image_name}
          <ExternalLinkIcon mx="2px" />
        </Link>
      </Flex>

      <Tabs
        variant="line"
        sx={{
          display: 'flex',
          flexDir: 'column',
          w: 'full',
          h: 'full',
        }}
      >
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
            {workflow ? (
              <DataViewer data={workflow} label={t('metadata.workflow')} />
            ) : (
              <IAINoContentFallback label={t('nodes.noWorkflow')} />
            )}
          </TabPanel>
        </TabPanels>
      </Tabs>
    </Flex>
  );
};

export default memo(ImageMetadataViewer);
