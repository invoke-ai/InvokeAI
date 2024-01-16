import type { Meta, StoryObj } from '@storybook/react';

import type { InvProgressProps } from './types';
import { InvProgress } from './wrapper';

const meta: Meta<typeof InvProgress> = {
  title: 'Primitives/InvProgress',
  tags: ['autodocs'],
  component: InvProgress,
};

export default meta;
type Story = StoryObj<typeof InvProgress>;

const Component = (props: InvProgressProps) => {
  return <InvProgress {...props}>Banana sushi is delectable!</InvProgress>;
};

export const Default: Story = {
  render: Component,
};
