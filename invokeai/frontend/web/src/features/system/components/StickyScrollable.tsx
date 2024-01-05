import { Flex } from '@chakra-ui/layout';
import { InvHeading } from 'common/components/InvHeading/wrapper';
import type { PropsWithChildren } from 'react';
import { memo } from 'react';

export type StickyScrollableHeadingProps = {
  title: string;
};

export const StickyScrollableHeading = memo(
  (props: StickyScrollableHeadingProps) => {
    return (
      <Flex ps={2} pb={4} position="sticky" zIndex={1} top={0} bg="base.800">
        <InvHeading size="sm">{props.title}</InvHeading>
      </Flex>
    );
  }
);

StickyScrollableHeading.displayName = 'StickyScrollableHeading';

export type StickyScrollableContentProps = PropsWithChildren;

export const StickyScrollableContent = memo(
  (props: StickyScrollableContentProps) => {
    return (
      <Flex p={4} borderRadius="base" bg="base.750" flexDir="column" gap={4}>
        {props.children}
      </Flex>
    );
  }
);

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
