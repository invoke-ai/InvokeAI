import type { Meta, StoryObj } from '@storybook/react';

import { ImageSettingsAccordion } from './ImageSettingsAccordion';

const meta: Meta<typeof ImageSettingsAccordion> = {
  title: 'Feature/ImageSettingsAccordion',
  tags: ['autodocs'],
  component: ImageSettingsAccordion,
};

export default meta;
type Story = StoryObj<typeof ImageSettingsAccordion>;

export const Default: Story = {
  render: () => <ImageSettingsAccordion />,
};
