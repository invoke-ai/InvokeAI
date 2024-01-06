import type { Meta, StoryObj } from '@storybook/react';

import { InvTextarea } from './InvTextarea';
import type { InvTextareaProps } from './types';

const meta: Meta<typeof InvTextarea> = {
  title: 'Primitives/InvTextarea',
  tags: ['autodocs'],
  component: InvTextarea,
};

export default meta;
type Story = StoryObj<typeof InvTextarea>;

const Component = (props: InvTextareaProps) => {
  return <InvTextarea {...props} />;
};

export const Default: Story = {
  render: Component,
};

export const Resizeable: Story = {
  render: Component,
  args: {
    resize: 'vertical',
    minW: '200px',
    minH: '50px',
  },
};
