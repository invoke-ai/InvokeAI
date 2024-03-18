import type { Meta, StoryObj } from '@storybook/react';

import { ControlSettingsAccordion } from './ControlSettingsAccordion';

const meta: Meta<typeof ControlSettingsAccordion> = {
  title: 'Feature/ControlSettingsAccordion',
  tags: ['autodocs'],
  component: ControlSettingsAccordion,
};

export default meta;
type Story = StoryObj<typeof ControlSettingsAccordion>;

export const Default: Story = {
  render: () => <ControlSettingsAccordion />,
};
