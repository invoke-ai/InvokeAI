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
import { memo, useMemo } from 'react';
import { useGetImageMetadataQuery } from 'services/api/endpoints/images';
import { ImageDTO } from 'services/api/types';
import { useDebounce } from 'use-debounce';
import ImageMetadataActions from './ImageMetadataActions';
import ImageMetadataJSON from './ImageMetadataJSON';

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

  const tabData = useMemo(() => {
    const _tabData: { label: string; data: object; copyTooltip: string }[] = [];

    if (metadata) {
      _tabData.push({
        label: 'Core Metadata',
        data: metadata,
        copyTooltip: 'Copy Core Metadata JSON',
      });
    }

    if (image) {
      _tabData.push({
        label: 'Image Details',
        data: image,
        copyTooltip: 'Copy Image Details JSON',
      });
    }

    if (graph) {
      _tabData.push({
        label: 'Graph',
        data: graph,
        copyTooltip: 'Copy Graph JSON',
      });
    }
    return _tabData;
  }, [metadata, graph, image]);

  return (
    <Flex
      sx={{
        padding: 4,
        gap: 1,
        flexDirection: 'column',
        width: 'full',
        height: 'full',
        backdropFilter: 'blur(20px)',
        bg: 'baseAlpha.200',
        _dark: {
          bg: 'blackAlpha.600',
        },
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
          {tabData.map((tab) => (
            <Tab
              key={tab.label}
              sx={{
                borderTopRadius: 'base',
              }}
            >
              <Text sx={{ color: 'base.700', _dark: { color: 'base.300' } }}>
                {tab.label}
              </Text>
            </Tab>
          ))}
        </TabList>

        <TabPanels sx={{ w: 'full', h: 'full' }}>
          {tabData.map((tab) => (
            <TabPanel
              key={tab.label}
              sx={{ w: 'full', h: 'full', p: 0, pt: 4 }}
            >
              <ImageMetadataJSON
                jsonObject={tab.data}
                copyTooltip={tab.copyTooltip}
              />
            </TabPanel>
          ))}
        </TabPanels>
      </Tabs>
    </Flex>
  );
};

export default memo(ImageMetadataViewer);
