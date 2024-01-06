import type { Meta, StoryObj } from '@storybook/react';
import { InvMenuItem } from 'common/components/InvMenu/InvMenuItem';
import { InvMenuList } from 'common/components/InvMenu/InvMenuList';
import { InvText } from 'common/components/InvText/wrapper';
import { useCallback } from 'react';
import { FaCopy, FaDownload, FaTrash } from 'react-icons/fa6';

import { InvContextMenu } from './InvContextMenu';

const meta: Meta<typeof InvContextMenu> = {
  title: 'Primitives/InvContextMenu',
  tags: ['autodocs'],
  component: InvContextMenu,
};

export default meta;
type Story = StoryObj<typeof InvContextMenu>;

const Component = () => {
  const renderMenuFunc = useCallback(
    () => (
      <InvMenuList>
        <InvMenuItem icon={<FaDownload />} command="⌘S">
          Download
        </InvMenuItem>
        <InvMenuItem icon={<FaCopy />} command="⌘C">
          Create a Copy
        </InvMenuItem>
        <InvMenuItem>Mark as Draft</InvMenuItem>
        <InvMenuItem icon={<FaTrash />} isDestructive>
          Delete
        </InvMenuItem>
        <InvMenuItem>Attend a Workshop</InvMenuItem>
      </InvMenuList>
    ),
    []
  );

  return (
    <InvContextMenu<HTMLParagraphElement> renderMenu={renderMenuFunc}>
      {(ref) => (
        <InvText ref={ref} p={5} bg="base.500">
          Right-click me
        </InvText>
      )}
    </InvContextMenu>
  );
};

export const Default: Story = {
  render: Component,
};
