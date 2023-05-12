import { Box, Flex, FormControl, FormLabel, Select } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAISlider from 'common/components/IAISlider';
import { setWidth } from 'features/parameters/store/generationSlice';
import { memo, useState } from 'react';
import AspectRatioPreview, {
  AspectRatio,
  Orientation,
} from './AspectRatioPreview';

const RATIOS: AspectRatio[] = [
  [1, 1],
  [5, 4],
  [3, 2],
  [16, 10],
  [16, 9],
];

RATIOS.forEach((r) => {
  const float = r[0] / r[1];
  console.log((512 * float) / 8);
});

const dimensionsSettingsSelector = createSelector(
  (state: RootState) => state.generation,
  (generation) => {
    const { width, height } = generation;

    return { width, height };
  }
);

const DimensionsSettings = () => {
  const { width, height } = useAppSelector(dimensionsSettingsSelector);
  const dispatch = useAppDispatch();
  const [ratioIndex, setRatioIndex] = useState(4);
  const [orientation, setOrientation] = useState<Orientation>('portrait');

  return (
    <Flex gap={3}>
      <Box flexShrink={0}>
        <AspectRatioPreview
          ratio={RATIOS[ratioIndex]}
          orientation={orientation}
          size="4rem"
        />
      </Box>
      <FormControl>
        <FormLabel>Aspect Ratio</FormLabel>
        <Select
          onChange={(e) => {
            setRatioIndex(Number(e.target.value));
          }}
        >
          {RATIOS.map((r, i) => (
            <option key={r.join()} value={i}>{`${r[0]}:${r[1]}`}</option>
          ))}
        </Select>
      </FormControl>
      <IAISlider
        label="Size"
        value={width}
        min={64}
        max={2048}
        step={8}
        onChange={(v) => {
          dispatch(setWidth(v));
        }}
      />
    </Flex>
  );
};

export default memo(DimensionsSettings);
