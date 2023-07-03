import { FaImages } from 'react-icons/fa';
import { Grid, GridItem } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import BatchImage from './BatchImage';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';

const selector = createSelector(
  stateSelector,
  (state) => {
    const imageNames = state.batch.imageNames.concat().reverse();

    return { imageNames };
  },
  defaultSelectorOptions
);

const BatchImageGrid = () => {
  const { imageNames } = useAppSelector(selector);

  if (imageNames.length === 0) {
    return (
      <IAINoContentFallback
        icon={FaImages}
        boxSize={16}
        label="No images in Batch"
      />
    );
  }

  return (
    <Grid
      sx={{
        position: 'absolute',
        flexWrap: 'wrap',
        w: 'full',
        minH: 0,
        maxH: 'full',
        overflowY: 'scroll',
        gridTemplateColumns: `repeat(auto-fill, minmax(128px, 1fr))`,
      }}
    >
      {imageNames.map((imageName) => (
        <GridItem key={imageName} sx={{ p: 1.5 }}>
          <BatchImage imageName={imageName} />
        </GridItem>
      ))}
    </Grid>
  );
};

export default BatchImageGrid;
