import { Flex } from '@chakra-ui/layout';
import type { Meta, StoryObj } from '@storybook/react';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvSlider } from 'common/components/InvSlider/InvSlider';
import { useState } from 'react';

import { AspectRatioPreview } from './AspectRatioPreview';

const meta: Meta<typeof AspectRatioPreview> = {
  title: 'Components/AspectRatioPreview',
  tags: ['autodocs'],
  component: AspectRatioPreview,
};

export default meta;
type Story = StoryObj<typeof InvControl>;

const MIN = 64;
const MAX = 1024;
const STEP = 64;
const FINE_STEP = 8;
const INITIAL = 512;
const MARKS = Array.from(
  { length: Math.floor(MAX / STEP) },
  (_, i) => MIN + i * STEP
);

const Component = () => {
  const [width, setWidth] = useState(INITIAL);
  const [height, setHeight] = useState(INITIAL);
  return (
    <Flex w="full" flexDir="column">
      <InvControl label="Width">
        <InvSlider
          value={width}
          min={MIN}
          max={MAX}
          step={STEP}
          fineStep={FINE_STEP}
          onChange={setWidth}
          marks={MARKS}
        />
      </InvControl>
      <InvControl label="Height">
        <InvSlider
          value={height}
          min={MIN}
          max={MAX}
          step={STEP}
          fineStep={FINE_STEP}
          onChange={setHeight}
          marks={MARKS}
        />
      </InvControl>
      <Flex h={96} w={96} p={4}>
        <AspectRatioPreview width={width} height={height} />
      </Flex>
    </Flex>
  );
};

export const AspectRatioWithSliderInvControls: Story = {
  render: Component,
};
