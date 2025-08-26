import { Flex, Grid, GridItem, Text } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { ASPECT_RATIO_MAP } from 'features/controlLayers/store/types';
import { useCurrentVideoDimensions } from 'features/parameters/hooks/useCurrentVideoDimensions';
import { selectVideoAspectRatio } from 'features/parameters/store/videoSlice';
import { memo, useMemo } from 'react';
import { useMeasure } from 'react-use';

export const VideoDimensionsPreview = memo(() => {
  const aspectRatio = useAppSelector(selectVideoAspectRatio);
  const [ref, dims] = useMeasure<HTMLDivElement>();

  const currentVideoDimensions = useCurrentVideoDimensions();

  const previewBoxSize = useMemo(() => {
    if (!dims) {
      return { width: 0, height: 0 };
    }

    const aspectRatioValue = ASPECT_RATIO_MAP[aspectRatio]?.ratio ?? 1;

    let width = currentVideoDimensions.width;
    let height = currentVideoDimensions.height;

    if (currentVideoDimensions.width > currentVideoDimensions.height) {
      width = dims.width;
      height = width / aspectRatioValue;
    } else {
      height = dims.height;
      width = height * aspectRatioValue;
    }

    return { width, height };
  }, [dims, currentVideoDimensions, aspectRatio]);

  return (
    <Flex w="full" h="full" alignItems="center" justifyContent="center" ref={ref}>
      <Flex
        position="relative"
        borderRadius="base"
        borderColor="base.600"
        borderWidth="3px"
        width={`${previewBoxSize.width}px`}
        height={`${previewBoxSize.height}px`}
        alignItems="center"
        justifyContent="center"
      >
        <Grid
          borderRadius="base"
          position="absolute"
          top={0}
          right={0}
          bottom={0}
          left={0}
          gridTemplateColumns="1fr 1fr 1fr"
          gridTemplateRows="1fr 1fr 1fr"
          gap="1px"
          bg="base.700"
        >
          <GridItem bg="base.800" />
          <GridItem bg="base.800" />
          <GridItem bg="base.800" />
          <GridItem bg="base.800" />
          <GridItem bg="base.800" />
          <GridItem bg="base.800" />
          <GridItem bg="base.800" />
          <GridItem bg="base.800" />
          <GridItem bg="base.800" />
        </Grid>
        <Flex
          position="absolute"
          top="50%"
          right="50%"
          bottom="50%"
          left="50%"
          alignItems="center"
          justifyContent="center"
        >
          <Text color="base.200" fontSize="xs">
            {currentVideoDimensions.width}x{currentVideoDimensions.height}
          </Text>
        </Flex>
      </Flex>
    </Flex>
  );
});

VideoDimensionsPreview.displayName = 'VideoDimensionsPreview';
