import { ChevronDownIcon } from '@chakra-ui/icons';
import type { Meta, StoryObj } from '@storybook/react';
import { InvButton } from 'common/components/InvButton/InvButton';
import { FaCopy, FaDownload, FaTrash } from 'react-icons/fa6';

import { InvMenuItem } from './InvMenuItem';
import { InvMenuList } from './InvMenuList';
import type { InvMenuProps } from './types';
import { InvMenu, InvMenuButton, InvMenuGroup } from './wrapper';

const meta: Meta<typeof InvMenu> = {
  title: 'Primitives/InvMenu',
  tags: ['autodocs'],
  component: InvMenu,
  args: {
    colorScheme: 'base',
  },
};

export default meta;
type Story = StoryObj<typeof InvMenu>;

const Component = (props: InvMenuProps) => {
  return (
    <InvMenu {...props}>
      <InvMenuButton as={InvButton} rightIcon={<ChevronDownIcon />}>
        Actions
      </InvMenuButton>
      <InvMenuList>
        <InvMenuGroup title="Some Category">
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
        </InvMenuGroup>
        <InvMenuGroup title="Another Category">
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
        </InvMenuGroup>
      </InvMenuList>
    </InvMenu>
  );
};

export const Default: Story = {
  render: Component,
};
