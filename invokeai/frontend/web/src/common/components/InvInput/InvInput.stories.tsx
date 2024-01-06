import type { Meta, StoryObj } from '@storybook/react';

import { InvInput } from './InvInput';
import type { InvInputProps } from './types';

const meta: Meta<typeof InvInput> = {
  title: 'Primitives/InvInput',
  tags: ['autodocs'],
  component: InvInput,
};

export default meta;
type Story = StoryObj<typeof InvInput>;

const Component = (props: InvInputProps) => {
  return <InvInput {...props} />;
};

export const Default: Story = {
  render: Component,
};
