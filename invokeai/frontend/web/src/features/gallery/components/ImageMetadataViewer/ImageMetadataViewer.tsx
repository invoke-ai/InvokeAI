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
import { memo } from 'react';
import { useGetImageMetadataQuery } from 'services/api/endpoints/images';
import { ImageDTO } from 'services/api/types';
import { useDebounce } from 'use-debounce';
import ImageMetadataActions from './ImageMetadataActions';
import DataViewer from './DataViewer';

type ImageMetadataViewerProps = {
  image: ImageDTO;
};

const ImageMetadataViewer = ({ image }: ImageMetadataViewerProps) => {
  // TODO: fix hotkeys
  // const dispatch = useAppDispatch();
  // useHotkeys('esc', () => {
  //   dispatch(setShouldShowImageDetails(false));
  // });

  const [debouncedMetadataQueryArg, debounceState] = useDebounce(
    image.image_name,
    500
  );

  const { currentData } = useGetImageMetadataQuery(
    debounceState.isPending()
      ? skipToken
      : debouncedMetadataQueryArg ?? skipToken
  );
  const metadata = currentData?.metadata;
  const graph = currentData?.graph;

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
          <Tab>Core Metadata</Tab>
          <Tab>Image Details</Tab>
          <Tab>Graph</Tab>
        </TabList>

        <TabPanels>
          <TabPanel>
            {metadata ? (
              <DataViewer data={metadata} label="Core Metadata" />
            ) : (
              <IAINoContentFallback label="No core metadata found" />
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
            {graph ? (
              <DataViewer data={graph} label="Graph" />
            ) : (
              <IAINoContentFallback label="No graph found" />
            )}
          </TabPanel>
        </TabPanels>
      </Tabs>
    </Flex>
  );
};

export default memo(ImageMetadataViewer);
