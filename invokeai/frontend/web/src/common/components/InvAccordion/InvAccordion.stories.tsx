import type { Meta, StoryObj } from '@storybook/react';
import { InvText } from 'common/components/InvText/wrapper';

import { InvAccordionButton } from './InvAccordionButton';
import type { InvAccordionProps } from './types';
import { InvAccordion, InvAccordionItem, InvAccordionPanel } from './wrapper';

const meta: Meta<typeof InvAccordion> = {
  title: 'Primitives/InvAccordion',
  tags: ['autodocs'],
  component: InvAccordion,
  args: {
    colorScheme: 'base',
  },
};

export default meta;
type Story = StoryObj<typeof InvAccordion>;

const Component = (props: InvAccordionProps) => {
  return (
    <InvAccordion {...props} defaultIndex={[0]} allowMultiple>
      <InvAccordionItem>
        <InvAccordionButton badges={['and', 'i', 'said']}>
          Section 1 title
        </InvAccordionButton>
        <InvAccordionPanel p={4}>
          <InvText>
            25 years and my life is still Tryin&apos; to get up that great big
            hill of hope For a destination I realized quickly when I knew I
            should That the world was made up of this brotherhood of man For
            whatever that means
          </InvText>
        </InvAccordionPanel>
      </InvAccordionItem>

      <InvAccordionItem>
        <InvAccordionButton badges={['heeeyyyyyy']}>
          Section 1 title
        </InvAccordionButton>
        <InvAccordionPanel p={4}>
          <InvText>
            And so I cry sometimes when I&apos;m lying in bed Just to get it all
            out what&apos;s in my head And I, I am feeling a little peculiar And
            so I wake in the morning and I step outside And I take a deep breath
            and I get real high And I scream from the top of my lungs
            &quot;What&apos;s going on?&quot;
          </InvText>
        </InvAccordionPanel>
      </InvAccordionItem>

      <InvAccordionItem>
        <InvAccordionButton badges={["what's", 'goin', 'on', '?']}>
          Section 2 title
        </InvAccordionButton>
        <InvAccordionPanel p={4}>
          <InvText>
            And I say, hey-ey-ey Hey-ey-ey I said &quot;Hey, a-what&apos;s going
            on?&quot; And I say, hey-ey-ey Hey-ey-ey I said &quot;Hey,
            a-what&apos;s going on?&quot;
          </InvText>
        </InvAccordionPanel>
      </InvAccordionItem>
    </InvAccordion>
  );
};

export const Default: Story = {
  render: Component,
};
