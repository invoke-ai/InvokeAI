import type { Meta, StoryObj } from '@storybook/react';

import type { InvCheckboxProps } from './types';
import { InvCheckbox } from './wrapper';

const meta: Meta<typeof InvCheckbox> = {
  title: 'Primitives/InvCheckbox',
  tags: ['autodocs'],
  component: InvCheckbox,
};

export default meta;
type Story = StoryObj<typeof InvCheckbox>;

const Component = (props: InvCheckboxProps) => {
  return <InvCheckbox {...props} />;
};

export const Default: Story = {
  render: Component,
};
