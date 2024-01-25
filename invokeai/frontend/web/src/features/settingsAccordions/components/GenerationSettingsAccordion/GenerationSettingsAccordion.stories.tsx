import type { Meta, StoryObj } from '@storybook/react';

import { GenerationSettingsAccordion } from './GenerationSettingsAccordion';

const meta: Meta<typeof GenerationSettingsAccordion> = {
  title: 'Feature/GenerationSettingsAccordion',
  tags: ['autodocs'],
  component: GenerationSettingsAccordion,
};

export default meta;
type Story = StoryObj<typeof GenerationSettingsAccordion>;

export const Default: Story = {
  render: () => <GenerationSettingsAccordion />,
};
