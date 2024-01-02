import type { Meta, StoryObj } from '@storybook/react';

import { Prompts } from './Prompts';

const meta: Meta<typeof Prompts> = {
  title: 'Feature/Prompt/Prompts',
  tags: ['autodocs'],
  component: Prompts,
};

export default meta;
type Story = StoryObj<typeof Prompts>;

const Component = () => {
  return <Prompts />;
};

export const Default: Story = {
  render: Component,
};
