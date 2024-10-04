import type { Meta, StoryObj } from '@storybook/react';

import QueueControls from './QueueControls';

const meta: Meta<typeof QueueControls> = {
  title: 'Feature/QueueControls',
  tags: ['autodocs'],
  component: QueueControls,
};

export default meta;
type Story = StoryObj<typeof QueueControls>;

export const Default: Story = {
  render: () => <QueueControls />,
};
