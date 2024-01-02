import type { Meta, StoryObj } from '@storybook/react';
import { FaBoltLightning } from 'react-icons/fa6';

import { InvIconButton } from './InvIconButton';
import type { InvIconButtonProps } from './types';

const meta: Meta<typeof InvIconButton> = {
  title: 'Primitives/InvIconButton',
  tags: ['autodocs'],
  component: InvIconButton,
  args: {
    icon: <FaBoltLightning />,
  },
};

export default meta;
type Story = StoryObj<typeof InvIconButton>;

const Component = (props: InvIconButtonProps) => {
  return <InvIconButton {...props} />;
};

export const Default: Story = {
  render: Component,
};
