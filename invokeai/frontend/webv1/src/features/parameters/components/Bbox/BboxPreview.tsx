import { Flex, Grid, GridItem } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { selectAspectRatioValue, selectHeight, selectWidth } from 'features/controlLayers/store/selectors';
import { memo, useMemo } from 'react';
import { useMeasure } from 'react-use';

export const BboxPreview = memo(() => {
  const bboxWidth = useAppSelector(selectWidth);
  const bboxHeight = useAppSelector(selectHeight);
  const aspectRatioValue = useAppSelector(selectAspectRatioValue);
  const [ref, dims] = useMeasure<HTMLDivElement>();

  const previewBoxSize = useMemo(() => {
    if (!dims) {
      return { width: 0, height: 0 };
    }

    let width = bboxWidth;
    let height = bboxHeight;

    if (bboxWidth > bboxHeight) {
      width = dims.width;
      height = width / aspectRatioValue;
    } else {
      height = dims.height;
      width = height * aspectRatioValue;
    }

    return { width, height };
  }, [dims, bboxWidth, bboxHeight, aspectRatioValue]);

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
      </Flex>
    </Flex>
  );
});

BboxPreview.displayName = 'BboxPreview';
