import type { Meta, StoryObj } from '@storybook/react';
import { InvNumberInput } from 'common/components/InvNumberInput/InvNumberInput';
import { InvSelect } from 'common/components/InvSelect/InvSelect';
import type { InvSelectOption } from 'common/components/InvSelect/types';
import { InvSlider } from 'common/components/InvSlider/InvSlider';
import { useState } from 'react';

import { InvControl } from './InvControl';
import type { InvControlProps } from './types';

const meta: Meta<typeof InvControl> = {
  title: 'Primitives/InvControl',
  tags: ['autodocs'],
  component: InvControl,
  args: {
    label: 'My Control',
    isDisabled: false,
    isInvalid: false,
    w: 96,
  },
};

export default meta;
type Story = StoryObj<typeof InvControl>;

const InvControlWithSliderComponent = (props: InvControlProps) => {
  const [value, setValue] = useState(0);
  return (
    <InvControl {...props}>
      <InvSlider value={value} min={0} max={10} step={1} onChange={setValue} />
    </InvControl>
  );
};

const InvControlWithSliderAndHelperTextComponent = (props: InvControlProps) => {
  const [value, setValue] = useState(0);
  return (
    <InvControl {...props} helperText="This is some helpful text">
      <InvSlider value={value} min={0} max={10} step={1} onChange={setValue} />
    </InvControl>
  );
};

const InvControlWithNumberInputComponent = (props: InvControlProps) => {
  const [value, setValue] = useState(0);
  return (
    <InvControl {...props}>
      <InvNumberInput
        value={value}
        min={0}
        max={10}
        step={1}
        onChange={setValue}
      />
    </InvControl>
  );
};

const options: InvSelectOption[] = [
  {
    value: 'chocolate',
    label: 'Chocolate',
  },
  {
    value: 'strawberry',
    label: 'Strawberry',
  },
  {
    value: 'vanilla',
    label: 'Vanilla',
  },
];
const InvControlWithSelectComponent = (props: InvControlProps) => {
  return (
    <InvControl {...props}>
      <InvSelect defaultValue={options[0]} options={options} />
    </InvControl>
  );
};

export const InvControlWithSlider: Story = {
  render: InvControlWithSliderComponent,
};

export const InvControlWithSliderAndHelperText: Story = {
  render: InvControlWithSliderAndHelperTextComponent,
};

export const InvControlWithNumberInput: Story = {
  render: InvControlWithNumberInputComponent,
};

export const InvControlWithSelect: Story = {
  render: InvControlWithSelectComponent,
};
