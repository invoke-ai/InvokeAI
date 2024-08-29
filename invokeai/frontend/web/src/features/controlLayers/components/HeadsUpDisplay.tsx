import { Grid, GridItem, Text } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { selectCanvasSlice } from 'features/controlLayers/store/selectors';
import { memo } from 'react';

const selectBbox = createSelector(selectCanvasSlice, (canvas) => canvas.bbox);

export const HeadsUpDisplay = memo(() => {
  const bbox = useAppSelector(selectBbox);

  return (
    <Grid
      bg="base.900"
      borderBottomEndRadius="base"
      p={2}
      gap={2}
      borderRadius="base"
      templateColumns="auto auto"
      opacity={0.6}
    >
      <HUDItem label="BBox" value={`${bbox.rect.width}×${bbox.rect.height} px`} />
      <HUDItem label="Scaled BBox" value={`${bbox.scaledSize.width}×${bbox.scaledSize.height} px`} />
    </Grid>
  );
});

HeadsUpDisplay.displayName = 'HeadsUpDisplay';

const HUDItem = memo(({ label, value }: { label: string; value: string | number }) => {
  return (
    <>
      <GridItem>
        <Text textAlign="end">{label}: </Text>
      </GridItem>
      <GridItem fontWeight="semibold">
        <Text>{value}</Text>
      </GridItem>
    </>
  );
});

HUDItem.displayName = 'HUDItem';
