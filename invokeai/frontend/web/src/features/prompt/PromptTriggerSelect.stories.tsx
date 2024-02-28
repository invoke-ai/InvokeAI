import type { Meta, StoryObj } from '@storybook/react';

import { PromptTriggerSelect } from './PromptTriggerSelect';
import type { PromptTriggerSelectProps } from './types';

const meta: Meta<typeof PromptTriggerSelect> = {
  title: 'Feature/Prompt/PromptTriggerSelect',
  tags: ['autodocs'],
  component: PromptTriggerSelect,
};

export default meta;
type Story = StoryObj<typeof PromptTriggerSelect>;

const Component = (props: PromptTriggerSelectProps) => {
  return <PromptTriggerSelect {...props}>Invoke</PromptTriggerSelect>;
};

export const Default: Story = {
  render: Component,
};
