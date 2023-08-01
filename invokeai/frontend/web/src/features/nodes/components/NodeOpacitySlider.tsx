import {
  Box,
  Slider,
  SliderFilledTrack,
  SliderThumb,
  SliderTrack,
} from '@chakra-ui/react';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useCallback } from 'react';
import { nodeOpacityChanged } from '../store/nodesSlice';

export default function NodeOpacitySlider() {
  const dispatch = useAppDispatch();
  const nodeOpacity = useAppSelector((state) => state.nodes.nodeOpacity);

  const handleChange = useCallback(
    (v: number) => {
      dispatch(nodeOpacityChanged(v));
    },
    [dispatch]
  );

  return (
    <Box>
      <Slider
        aria-label="Node Opacity"
        value={nodeOpacity}
        min={0.5}
        max={1}
        step={0.01}
        onChange={handleChange}
        orientation="vertical"
        defaultValue={30}
      >
        <SliderTrack>
          <SliderFilledTrack />
        </SliderTrack>
        <SliderThumb />
      </Slider>
    </Box>
  );
}
