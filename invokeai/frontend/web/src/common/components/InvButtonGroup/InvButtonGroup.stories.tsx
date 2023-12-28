import type { Meta, StoryObj } from '@storybook/react';
import { InvButton } from 'common/components/InvButton/InvButton';
import { InvIconButton } from 'common/components/InvIconButton/InvIconButton';
import { FaImage } from 'react-icons/fa';

import { InvButtonGroup } from './InvButtonGroup';
import type { InvButtonGroupProps } from './types';

const meta: Meta<typeof InvButtonGroup> = {
  title: 'Primitives/InvButtonGroup',
  tags: ['autodocs'],
  component: InvButtonGroup,
  args: {
    colorScheme: 'base',
  },
};

export default meta;
type Story = StoryObj<typeof InvButtonGroup>;

const Component = (props: InvButtonGroupProps) => {
  return (
    <InvButtonGroup {...props}>
      <InvButton>Test</InvButton>
      <InvButton>Test</InvButton>
      <InvButton>Test</InvButton>
    </InvButtonGroup>
  );
};

const ComponentWithIconButtons = (props: InvButtonGroupProps) => {
  return (
    <InvButtonGroup {...props}>
      <InvIconButton aria-label="test" icon={<FaImage />} />
      <InvIconButton aria-label="test" icon={<FaImage />} />
      <InvIconButton aria-label="test" icon={<FaImage />} />
    </InvButtonGroup>
  );
};

export const Default: Story = {
  render: Component,
};

export const WithIconButtons: Story = {
  render: ComponentWithIconButtons,
};
