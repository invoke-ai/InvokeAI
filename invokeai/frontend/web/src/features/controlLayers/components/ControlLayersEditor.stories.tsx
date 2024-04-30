import { Flex } from '@invoke-ai/ui-library';
import type { Meta, StoryObj } from '@storybook/react';
import { ControlLayersEditor } from 'features/controlLayers/components/ControlLayersEditor';

const meta: Meta<typeof ControlLayersEditor> = {
  title: 'Feature/ControlLayers',
  tags: ['autodocs'],
  component: ControlLayersEditor,
};

export default meta;
type Story = StoryObj<typeof ControlLayersEditor>;

const Component = () => {
  return (
    <Flex w={1500} h={1500}>
      <ControlLayersEditor />
    </Flex>
  );
};

export const Default: Story = {
  render: Component,
};
