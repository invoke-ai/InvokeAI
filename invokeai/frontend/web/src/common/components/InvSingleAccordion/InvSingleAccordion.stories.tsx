import type { Meta, StoryObj } from '@storybook/react';
import { InvTab } from 'common/components/InvTabs/InvTab';
import {
  InvTabList,
  InvTabPanel,
  InvTabPanels,
  InvTabs,
} from 'common/components/InvTabs/wrapper';
import { InvText } from 'common/components/InvText/wrapper';

import { InvSingleAccordion } from './InvSingleAccordion';
import type { InvSingleAccordionProps } from './types';

const meta: Meta<typeof InvSingleAccordion> = {
  title: 'Primitives/InvSingleAccordion',
  tags: ['autodocs'],
  component: InvSingleAccordion,
};

export default meta;
type Story = StoryObj<typeof InvSingleAccordion>;

const Component = (props: InvSingleAccordionProps) => {
  return (
    <InvSingleAccordion
      {...props}
      label="The Best Flavours of Banana Sushi"
      badges={['Yum', 'Gourmet', 'Barf']}
      defaultIsOpen
    >
      <InvTabs variant="collapse">
        <InvTabList>
          <InvTab>Caramelized</InvTab>
          <InvTab badges={[2]}>Peanut Butter</InvTab>
          <InvTab badges={[4]}>Chocolate-Dipped</InvTab>
        </InvTabList>

        <InvTabPanels>
          <InvTabPanel>
            <InvText>
              Slices of banana are caramelized with brown sugar and butter, then
              rolled in sushi rice and topped with a drizzle of caramel sauce.
              This variety offers a sweet and rich flavor, combining the
              creaminess of banana with the indulgent taste of caramel.
            </InvText>
          </InvTabPanel>
          <InvTabPanel>
            <InvText>
              A combination of creamy peanut butter and ripe banana slices,
              wrapped in sushi rice and seaweed. This sushi delivers a
              satisfying balance of nutty and sweet flavors, appealing to those
              who enjoy classic peanut butter and banana pairings.
            </InvText>
          </InvTabPanel>
          <InvTabPanel>
            <InvText>
              Banana slices are dipped in melted dark chocolate, then rolled in
              sushi rice and sprinkled with toasted sesame seeds. This type
              provides a decadent chocolate experience with a hint of nuttiness
              and the natural sweetness of banana.
            </InvText>
          </InvTabPanel>
        </InvTabPanels>
      </InvTabs>
    </InvSingleAccordion>
  );
};

export const Default: Story = {
  render: Component,
};
