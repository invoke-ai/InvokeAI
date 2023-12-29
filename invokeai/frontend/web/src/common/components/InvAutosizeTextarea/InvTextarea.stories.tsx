import type { Meta, StoryObj } from '@storybook/react';

import { InvAutosizeTextarea } from './InvAutosizeTextarea';
import type { InvAutosizeTextareaProps } from './types';

const meta: Meta<typeof InvAutosizeTextarea> = {
  title: 'Primitives/InvAutosizeTextarea',
  tags: ['autodocs'],
  component: InvAutosizeTextarea,
};

export default meta;
type Story = StoryObj<typeof InvAutosizeTextarea>;

const Component = (props: InvAutosizeTextareaProps) => {
  return <InvAutosizeTextarea {...props} />;
};

export const Default: Story = {
  render: Component,
};
