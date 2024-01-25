import type { Meta, StoryObj } from '@storybook/react';

import { RefinerSettingsAccordion } from './RefinerSettingsAccordion';

const meta: Meta<typeof RefinerSettingsAccordion> = {
  title: 'Feature/RefinerSettingsAccordion',
  tags: ['autodocs'],
  component: RefinerSettingsAccordion,
};

export default meta;
type Story = StoryObj<typeof RefinerSettingsAccordion>;

export const Default: Story = {
  render: () => <RefinerSettingsAccordion />,
};
