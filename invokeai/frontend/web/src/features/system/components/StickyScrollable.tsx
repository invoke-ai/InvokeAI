import { Flex, Heading } from '@invoke-ai/ui-library';
import type { PropsWithChildren } from 'react';
import { memo } from 'react';

type StickyScrollableHeadingProps = {
  title: string;
};

const StickyScrollableHeading = memo((props: StickyScrollableHeadingProps) => {
  return (
    <Flex ps={2} pb={4} position="sticky" zIndex={1} top={0} bg="base.800">
      <Heading size="sm">{props.title}</Heading>
    </Flex>
  );
});

StickyScrollableHeading.displayName = 'StickyScrollableHeading';

type StickyScrollableContentProps = PropsWithChildren;

const StickyScrollableContent = memo((props: StickyScrollableContentProps) => {
  return (
    <Flex p={4} borderRadius="base" bg="base.750" flexDir="column" gap={4}>
      {props.children}
    </Flex>
  );
});

StickyScrollableContent.displayName = 'StickyScrollableContent';

export type StickyScrollableProps = PropsWithChildren<{
  title: string;
}>;

export const StickyScrollable = memo((props: StickyScrollableProps) => {
  return (
    <Flex key={props.title} flexDir="column">
      <StickyScrollableHeading title={props.title} />
      <StickyScrollableContent>{props.children}</StickyScrollableContent>
    </Flex>
  );
});

StickyScrollable.displayName = 'StickyScrollable';
