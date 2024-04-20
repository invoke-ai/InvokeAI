import { Flex } from '@invoke-ai/ui-library';
import type { Meta, StoryObj } from '@storybook/react';
import { RegionalPromptsEditor } from 'features/regionalPrompts/components/RegionalPromptsEditor';

const meta: Meta<typeof RegionalPromptsEditor> = {
  title: 'Feature/RegionalPrompts',
  tags: ['autodocs'],
  component: RegionalPromptsEditor,
};

export default meta;
type Story = StoryObj<typeof RegionalPromptsEditor>;

const Component = () => {
  return (
    <Flex w={1500} h={1500}>
      <RegionalPromptsEditor />
    </Flex>
  );
};

export const Default: Story = {
  render: Component,
};
