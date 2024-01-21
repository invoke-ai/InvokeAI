import type { Meta, StoryObj } from '@storybook/react';

import { CompositingSettingsAccordion } from './CompositingSettingsAccordion';

const meta: Meta<typeof CompositingSettingsAccordion> = {
  title: 'Feature/CompositingSettingsAccordion',
  tags: ['autodocs'],
  component: CompositingSettingsAccordion,
};

export default meta;
type Story = StoryObj<typeof CompositingSettingsAccordion>;

export const Default: Story = {
  render: () => <CompositingSettingsAccordion />,
};
