import type { Meta, StoryObj } from '@storybook/react';

import type { InvSkeletonProps } from './types';
import { InvSkeleton } from './wrapper';

const meta: Meta<typeof InvSkeleton> = {
  title: 'Primitives/InvSkeleton',
  tags: ['autodocs'],
  component: InvSkeleton,
};

export default meta;
type Story = StoryObj<typeof InvSkeleton>;

const Component = (props: InvSkeletonProps) => {
  return <InvSkeleton {...props}>Banana sushi is delectable!</InvSkeleton>;
};

export const Default: Story = {
  render: Component,
};
