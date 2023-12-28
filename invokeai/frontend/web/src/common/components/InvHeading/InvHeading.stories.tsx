import type { Meta, StoryObj } from '@storybook/react';

import type { InvHeadingProps } from './types';
import { InvHeading } from './wrapper';

const meta: Meta<typeof InvHeading> = {
  title: 'Primitives/InvHeading',
  tags: ['autodocs'],
  component: InvHeading,
};

export default meta;
type Story = StoryObj<typeof InvHeading>;

const Component = (props: InvHeadingProps) => {
  return <InvHeading {...props}>Banana sushi is delectable!</InvHeading>;
};

export const Default: Story = {
  render: Component,
};
