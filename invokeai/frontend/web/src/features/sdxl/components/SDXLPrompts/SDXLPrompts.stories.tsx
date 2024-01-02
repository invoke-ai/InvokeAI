import type { Meta, StoryObj } from '@storybook/react';

import { SDXLPrompts } from './SDXLPrompts';

const meta: Meta<typeof SDXLPrompts> = {
  title: 'Feature/Prompt/SDXLPrompts',
  tags: ['autodocs'],
  component: SDXLPrompts,
};

export default meta;
type Story = StoryObj<typeof SDXLPrompts>;

const Component = () => {
  return <SDXLPrompts />;
};

export const Default: Story = {
  render: Component,
};
