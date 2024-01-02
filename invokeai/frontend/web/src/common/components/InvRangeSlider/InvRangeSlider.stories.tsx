import type { Meta, StoryObj } from '@storybook/react';
import { InvControl } from 'common/components/InvControl/InvControl';
import type { InvRangeSliderProps } from 'common/components/InvRangeSlider/types';
import { useCallback, useState } from 'react';

import { InvRangeSlider } from './InvRangeSlider';

const meta: Meta<typeof InvRangeSlider> = {
  title: 'Primitives/InvRangeSlider',
  tags: ['autodocs'],
  component: InvRangeSlider,
  args: {
    min: 0,
    max: 10,
    step: 1,
    marks: [0, 5, 10],
  },
};

export default meta;
type Story = StoryObj<typeof InvRangeSlider>;

const Component = (props: InvRangeSliderProps) => {
  const [value, setValue] = useState<[number, number]>([2, 8]);
  const onReset = useCallback(() => {
    setValue([2, 8]);
  }, []);
  const onChange = useCallback((v: [number, number]) => {
    setValue(v);
  }, []);
  return (
    <InvControl label="Slider">
      <InvRangeSlider
        {...props}
        value={value}
        onChange={onChange}
        onReset={onReset}
      />
    </InvControl>
  );
};

export const Default: Story = {
  render: Component,
  args: {
    fineStep: 0.1,
    withThumbTooltip: true,
    formatValue: (v: number) => `${v} eggs`,
  },
};
