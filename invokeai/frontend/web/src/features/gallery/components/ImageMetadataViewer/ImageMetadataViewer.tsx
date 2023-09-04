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
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import { memo } from 'react';
import { useGetImageMetadataFromFileQuery } from 'services/api/endpoints/images';
import { ImageDTO } from 'services/api/types';
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

  const { metadata, workflow } = useGetImageMetadataFromFileQuery(image, {
    selectFromResult: (res) => ({
      metadata: res?.currentData?.metadata,
      workflow: res?.currentData?.workflow,
    }),
  });

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

      <ImageMetadataActions metadata={metadata} />

      <Tabs
        variant="line"
        sx={{ display: 'flex', flexDir: 'column', w: 'full', h: 'full' }}
      >
        <TabList>
          <Tab>Metadata</Tab>
          <Tab>Image Details</Tab>
          <Tab>Workflow</Tab>
        </TabList>

        <TabPanels>
          <TabPanel>
            {metadata ? (
              <DataViewer data={metadata} label="Metadata" />
            ) : (
              <IAINoContentFallback label="No metadata found" />
            )}
          </TabPanel>
          <TabPanel>
            {image ? (
              <DataViewer data={image} label="Image Details" />
            ) : (
              <IAINoContentFallback label="No image details found" />
            )}
          </TabPanel>
          <TabPanel>
            {workflow ? (
              <DataViewer data={workflow} label="Workflow" />
            ) : (
              <IAINoContentFallback label="No workflow found" />
            )}
          </TabPanel>
        </TabPanels>
      </Tabs>
    </Flex>
  );
};

export default memo(ImageMetadataViewer);
