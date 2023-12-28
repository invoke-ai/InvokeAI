import type { Meta, StoryObj } from '@storybook/react';

import { AdvancedSettingsAccordion } from './AdvancedSettingsAccordion';

const meta: Meta<typeof AdvancedSettingsAccordion> = {
  title: 'Feature/AdvancedSettingsAccordion',
  tags: ['autodocs'],
  component: AdvancedSettingsAccordion,
};

export default meta;
type Story = StoryObj<typeof AdvancedSettingsAccordion>;

export const Default: Story = {
  render: () => <AdvancedSettingsAccordion />,
};
