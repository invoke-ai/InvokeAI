import type { Meta, StoryObj } from '@storybook/react';

import { EmbeddingSelect } from './EmbeddingSelect';
import type { EmbeddingSelectProps } from './types';

const meta: Meta<typeof EmbeddingSelect> = {
  title: 'Feature/Prompt/EmbeddingSelect',
  tags: ['autodocs'],
  component: EmbeddingSelect,
};

export default meta;
type Story = StoryObj<typeof EmbeddingSelect>;

const Component = (props: EmbeddingSelectProps) => {
  return <EmbeddingSelect {...props}>Invoke</EmbeddingSelect>;
};

export const Default: Story = {
  render: Component,
};
