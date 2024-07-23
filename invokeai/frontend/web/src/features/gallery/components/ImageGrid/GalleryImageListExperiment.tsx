import { Box, Flex, Image, Skeleton, Text } from '@invoke-ai/ui-library';
import { useAppStore } from 'app/store/storeHooks';
import { overlayScrollbarsParams } from 'common/components/OverlayScrollbars/constants';
import { debounce } from 'lodash-es';
import { useOverlayScrollbars } from 'overlayscrollbars-react';
import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { ListRange } from 'react-virtuoso';
import { Virtuoso } from 'react-virtuoso';
import { imagesApi, useGetImageNamesQuery, useLazyGetImagesByNameQuery } from 'services/api/endpoints/images';
import type { ImageDTO } from 'services/api/types';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type TableVirtuosoScrollerRef = (ref: HTMLElement | Window | null) => any;

export const GalleryImageListExperiment = memo(() => {
  const store = useAppStore();
  const { data } = useGetImageNamesQuery({ starred_first: false });
  const [getImagesByName] = useLazyGetImagesByNameQuery();

  const itemContent = useCallback((index: number, data: string) => {
    return <ListItem index={index} data={data} />;
  }, []);

  const onRangeChanged = useCallback(
    ({ startIndex, endIndex }: ListRange) => {
      // user has scrolled to a new range, fetch images that are not already in the store
      console.log('rangeChanged', startIndex, endIndex);

      // get the list of image names represented by this range
      // endIndex must be +1 bc else we miss the last image
      const imageNames = data?.slice(startIndex, endIndex + 1);

      if (imageNames) {
        // optimisation: we may have already loaded some of these images, so filter out the ones we already have
        const imageNamesToFetch: string[] = [];
        for (const name of imageNames) {
          // check if we have this image cached already
          const { data } = imagesApi.endpoints.getImageDTO.select(name)(store.getState());
          if (!data) {
            // nope, we need to fetch it
            imageNamesToFetch.push(name);
          }
        }
        console.log('imageNamesToFetch', imageNamesToFetch);
        getImagesByName({ image_names: imageNamesToFetch });
      }
    },
    [data, getImagesByName, store]
  );

  // debounce the onRangeChanged callback to avoid fetching images too frequently
  const debouncedOnRangeChanged = useMemo(() => debounce(onRangeChanged, 300), [onRangeChanged]);

  const rootRef = useRef<HTMLDivElement>(null);
  const [scroller, setScroller] = useState<HTMLElement | null>(null);
  const [initialize, osInstance] = useOverlayScrollbars(overlayScrollbarsParams);

  useEffect(() => {
    const { current: root } = rootRef;
    if (scroller && root) {
      initialize({
        target: root,
        elements: {
          viewport: scroller,
        },
      });
    }
    return () => osInstance()?.destroy();
  }, [scroller, initialize, osInstance]);

  if (!data) {
    return null;
  }

  return (
    <Box ref={rootRef} position="relative" w="full" h="full" mt={2}>
      <Virtuoso
        data={data}
        itemContent={itemContent}
        rangeChanged={debouncedOnRangeChanged}
        // increases teh virual viewport by 200px in each direction, so we fetch a few more images than required
        increaseViewportBy={200}
        scrollerRef={setScroller as TableVirtuosoScrollerRef}
      />
    </Box>
  );
});

GalleryImageListExperiment.displayName = 'GalleryImageListExperiment';

const useGetImageDTOCache = (imageName: string): ImageDTO | undefined => {
  // get the image data for this image - useQueryState does not trigger a fetch
  const { data, isUninitialized } = imagesApi.endpoints.getImageDTO.useQueryState(imageName);
  // but we want this component to be a subscriber of the cache! that way, when this component unmounts, the query cache is automatically cleared
  // useQuerySubscription allows us to subscribe, but by default it fetches the data immediately. using skip we can prevent that
  // the result is we never fetch data for this image from this component, it only subscribes to the cache
  // unfortunately this subcribe-to-cache-but-don't-fetch functionality is not built in to RTKQ.
  imagesApi.endpoints.getImageDTO.useQuerySubscription(imageName, { skip: isUninitialized });

  return data;
};

// the skeleton and real component need to be the same size else virtuoso will need to call rangeChanged multiples times to fill
const HEIGHT = 24;

const ListItem = ({ index, data }: { index: number; data: string }) => {
  const imageDTO = useGetImageDTOCache(data);

  if (!imageDTO) {
    return <Skeleton w="full" h={HEIGHT} />;
  }

  return (
    <Flex h={HEIGHT}>
      <Image src={imageDTO.thumbnail_url} h="full" aspectRatio="1/1" />
      <Flex flexDir="column">
        <Text>{index}</Text>
        <Text>{imageDTO.image_name}</Text>
      </Flex>
    </Flex>
  );
};
