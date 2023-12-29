import type { Meta, StoryObj } from '@storybook/react';
import { useState } from 'react';

import { InvNumberInput } from './InvNumberInput';
import type { InvNumberInputProps } from './types';

const meta: Meta<typeof InvNumberInput> = {
  title: 'Primitives/InvNumberInput',
  tags: ['autodocs'],
  component: InvNumberInput,
  args: {
    min: -10,
    max: 10,
    step: 1,
  },
};

export default meta;
type Story = StoryObj<typeof InvNumberInput>;

const Component = (props: InvNumberInputProps) => {
  const [value, setValue] = useState(0);
  return <InvNumberInput {...props} value={value} onChange={setValue} />;
};

export const Default: Story = {
  render: Component,
  args: { fineStep: 0.1 },
};

export const Integer: Story = {
  render: Component,
};
