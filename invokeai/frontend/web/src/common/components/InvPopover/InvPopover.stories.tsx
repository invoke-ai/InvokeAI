import type { Meta, StoryObj } from '@storybook/react';
import { InvButton } from 'common/components/InvButton/InvButton';

import {
  InvPopover,
  InvPopoverArrow,
  InvPopoverBody,
  InvPopoverCloseButton,
  InvPopoverContent,
  InvPopoverHeader,
  InvPopoverTrigger,
} from './wrapper';

const meta: Meta<typeof InvPopover> = {
  title: 'Primitives/InvPopover',
  tags: ['autodocs'],
  component: InvPopover,
  args: {
    colorScheme: 'base',
  },
};

export default meta;
type Story = StoryObj<typeof InvPopover>;

const Component = () => {
  return (
    <InvPopover>
      <InvPopoverTrigger>
        <InvButton>Trigger</InvButton>
      </InvPopoverTrigger>
      <InvPopoverContent>
        <InvPopoverArrow />
        <InvPopoverCloseButton />
        <InvPopoverHeader>Confirmation!</InvPopoverHeader>
        <InvPopoverBody>
          Are you sure you want to have that milkshake?
        </InvPopoverBody>
      </InvPopoverContent>
    </InvPopover>
  );
};

export const Default: Story = {
  render: Component,
};
