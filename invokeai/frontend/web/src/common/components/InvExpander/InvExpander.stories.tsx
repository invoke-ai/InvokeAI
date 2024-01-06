import type { Meta, StoryObj } from '@storybook/react';

import { InvExpander } from './InvExpander';
import type { InvExpanderProps } from './types';

const meta: Meta<typeof InvExpander> = {
  title: 'Primitives/InvExpander',
  tags: ['autodocs'],
  component: InvExpander,
};

export default meta;
type Story = StoryObj<typeof InvExpander>;

const Component = (props: InvExpanderProps) => {
  return <InvExpander {...props}>Invoke</InvExpander>;
};

export const Default: Story = {
  render: Component,
};
