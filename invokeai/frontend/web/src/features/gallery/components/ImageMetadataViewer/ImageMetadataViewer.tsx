import {
  ExternalLink,
  Flex,
  IconButton,
  Input,
  InputGroup,
  InputRightElement,
  Tab,
  TabList,
  TabPanel,
  TabPanels,
  Tabs,
} from '@invoke-ai/ui-library';
import { IAINoContentFallback, IAINoContentFallbackWithSpinner } from 'common/components/IAIImageFallback';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import ImageMetadataGraphTabContent from 'features/gallery/components/ImageMetadataViewer/ImageMetadataGraphTabContent';
import { ImageMetadataHandlers } from 'features/metadata/parsing';
import type { ChangeEvent } from 'react';
import { memo, useCallback, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { PiXBold } from 'react-icons/pi';
import { useDebouncedMetadata } from 'services/api/hooks/useDebouncedMetadata';
import type { ImageDTO } from 'services/api/types';

import DataViewer from './DataViewer';
import { ImageMetadataActions, UnrecallableMetadataDatum } from './ImageMetadataActions';
import ImageMetadataWorkflowTabContent from './ImageMetadataWorkflowTabContent';

type ImageMetadataViewerProps = {
  image: ImageDTO;
};

const CODE_TAB_PADDING_INLINE = 18;
const TAB_INDEX = {
  recall: 0,
  metadata: 1,
  imageDetails: 2,
  workflow: 3,
  graph: 4,
} as const;
const TAB_COUNT = Object.keys(TAB_INDEX).length;

const ImageMetadataViewer = ({ image }: ImageMetadataViewerProps) => {
  // TODO: fix hotkeys
  // const dispatch = useAppDispatch();
  // useHotkeys('esc', () => {
  //   dispatch(setShouldShowImageDetails(false));
  // });
  const { t } = useTranslation();

  const { metadata, isLoading } = useDebouncedMetadata(image.image_name);
  const [activeTabIndex, setActiveTabIndex] = useState(0);
  const [searchTerms, setSearchTerms] = useState<string[]>(() => Array(TAB_COUNT).fill(''));
  const isSearchableTab = activeTabIndex !== TAB_INDEX.recall;
  const activeSearchTerm = searchTerms[activeTabIndex] ?? '';

  const handleTabChange = useCallback((index: number) => {
    setActiveTabIndex(index);
  }, []);

  const handleChangeSearch = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      const value = e.target.value;
      setSearchTerms((prev) => {
        const next = [...prev];
        next[activeTabIndex] = value;
        return next;
      });
    },
    [activeTabIndex]
  );

  const handleClearSearch = useCallback(() => {
    setSearchTerms((prev) => {
      const next = [...prev];
      next[activeTabIndex] = '';
      return next;
    });
  }, [activeTabIndex]);

  return (
    <Flex
      layerStyle="first"
      padding={4}
      paddingInline={16}
      gap={1}
      flexDirection="column"
      width="full"
      height="full"
      borderRadius="base"
      position="absolute"
      overflow="hidden"
    >
      <ExternalLink href={image.image_url} label={image.image_name} />
      <UnrecallableMetadataDatum metadata={metadata} handler={ImageMetadataHandlers.CreatedBy} />

      <Tabs
        variant="line"
        isLazy={true}
        display="flex"
        flexDir="column"
        w="full"
        h="full"
        index={activeTabIndex}
        onChange={handleTabChange}
      >
        <Flex alignItems="flex-start" gap={2} borderBottomWidth="1px" borderColor="base.600">
          <TabList flex="1" pb={2} borderBottom="none">
            <Tab>{t('metadata.recallParameters')}</Tab>
            <Tab>{t('metadata.metadata')}</Tab>
            <Tab>{t('metadata.imageDetails')}</Tab>
            <Tab>{t('metadata.workflow')}</Tab>
            <Tab>{t('nodes.graph')}</Tab>
          </TabList>
          {isSearchableTab && (
            <InputGroup size="sm" w={48} me={6}>
              <Input placeholder={t('common.search')} value={activeSearchTerm} onChange={handleChangeSearch} />
              {activeSearchTerm && (
                <InputRightElement h="full" pe={2}>
                  <IconButton
                    aria-label={t('boards.clearSearch')}
                    icon={<PiXBold size={16} />}
                    variant="link"
                    opacity={0.7}
                    onClick={handleClearSearch}
                    size="sm"
                  />
                </InputRightElement>
              )}
            </InputGroup>
          )}
        </Flex>

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
              <Flex w="full" h="full" paddingInline={CODE_TAB_PADDING_INLINE}>
                <DataViewer
                  fileName={`${image.image_name.replace('.png', '')}_metadata`}
                  data={metadata}
                  label={t('metadata.metadata')}
                  withSearch
                  searchTerm={searchTerms[TAB_INDEX.metadata]}
                  showSearchInput={false}
                />
              </Flex>
            ) : (
              <IAINoContentFallback label={t('metadata.noMetaData')} />
            )}
          </TabPanel>
          <TabPanel>
            {image ? (
              <Flex w="full" h="full" paddingInline={CODE_TAB_PADDING_INLINE}>
                <DataViewer
                  fileName={`${image.image_name.replace('.png', '')}_details`}
                  data={image}
                  label={t('metadata.imageDetails')}
                  withSearch
                  searchTerm={searchTerms[TAB_INDEX.imageDetails]}
                  showSearchInput={false}
                />
              </Flex>
            ) : (
              <IAINoContentFallback label={t('metadata.noImageDetails')} />
            )}
          </TabPanel>
          <TabPanel>
            <Flex w="full" h="full" paddingInline={CODE_TAB_PADDING_INLINE}>
              <ImageMetadataWorkflowTabContent
                image={image}
                searchTerm={searchTerms[TAB_INDEX.workflow]}
                showSearchInput={false}
              />
            </Flex>
          </TabPanel>
          <TabPanel>
            <Flex w="full" h="full" paddingInline={CODE_TAB_PADDING_INLINE}>
              <ImageMetadataGraphTabContent
                image={image}
                searchTerm={searchTerms[TAB_INDEX.graph]}
                showSearchInput={false}
              />
            </Flex>
          </TabPanel>
        </TabPanels>
      </Tabs>
    </Flex>
  );
};

export default memo(ImageMetadataViewer);
