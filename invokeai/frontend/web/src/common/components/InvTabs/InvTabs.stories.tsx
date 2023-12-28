import type { Meta, StoryObj } from '@storybook/react';

import { InvTab } from './InvTab';
import type { InvTabsProps } from './types';
import { InvTabList, InvTabPanel, InvTabPanels, InvTabs } from './wrapper';

const meta: Meta<typeof InvTabs> = {
  title: 'Primitives/InvTabs',
  tags: ['autodocs'],
  component: InvTabs,
  args: {
    colorScheme: 'base',
    variant: 'collapse',
  },
};

export default meta;
type Story = StoryObj<typeof InvTabs>;

const Component = (props: InvTabsProps) => {
  return (
    <InvTabs {...props}>
      <InvTabList>
        <InvTab>One</InvTab>
        <InvTab>Two</InvTab>
        <InvTab>Three</InvTab>
      </InvTabList>

      <InvTabPanels>
        <InvTabPanel>
          <p>one!</p>
        </InvTabPanel>
        <InvTabPanel>
          <p>two!</p>
        </InvTabPanel>
        <InvTabPanel>
          <p>three!</p>
        </InvTabPanel>
      </InvTabPanels>
    </InvTabs>
  );
};

export const Default: Story = {
  render: Component,
};
