import type { Meta, StoryObj } from '@storybook/react';

import type { InvCardProps } from './types';
import { InvCard } from './wrapper';

const meta: Meta<typeof InvCard> = {
  title: 'Primitives/InvCard',
  tags: ['autodocs'],
  component: InvCard,
  args: {
    colorScheme: 'base',
  },
};

export default meta;
type Story = StoryObj<typeof InvCard>;

const Component = (props: InvCardProps) => {
  return <InvCard {...props}>Invoke</InvCard>;
};

export const Default: Story = {
  render: Component,
};
