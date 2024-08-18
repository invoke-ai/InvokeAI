import { Flex } from '@invoke-ai/ui-library';
import type { Meta, StoryObj } from '@storybook/react';
import { CanvasEditor } from 'features/controlLayers/components/ControlLayersEditor';

const meta: Meta<typeof CanvasEditor> = {
  title: 'Feature/ControlLayers',
  tags: ['autodocs'],
  component: CanvasEditor,
};

export default meta;
type Story = StoryObj<typeof CanvasEditor>;

const Component = () => {
  return (
    <Flex w={1500} h={1500}>
      <CanvasEditor />
    </Flex>
  );
};

export const Default: Story = {
  render: Component,
};
