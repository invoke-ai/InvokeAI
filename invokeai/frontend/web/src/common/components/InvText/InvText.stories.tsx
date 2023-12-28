import type { Meta, StoryObj } from '@storybook/react';

import type { InvTextProps } from './types';
import { InvText } from './wrapper';

const meta: Meta<typeof InvText> = {
  title: 'Primitives/InvText',
  tags: ['autodocs'],
  component: InvText,
};

export default meta;
type Story = StoryObj<typeof InvText>;

const Component = (props: InvTextProps) => {
  return <InvText {...props}>Banana sushi is delectable!</InvText>;
};

export const Default: Story = {
  render: Component,
};
