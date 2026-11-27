import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Flex, Heading } from '@invoke-ai/ui-library';
import type { PropsWithChildren } from 'react';
import { memo } from 'react';

type StickyScrollableHeadingProps = {
  title: string;
  sx?: SystemStyleObject;
};

const StickyScrollableHeading = memo((props: StickyScrollableHeadingProps) => {
  return (
    <Flex ps={2} pb={4} position="sticky" zIndex={1} top={0} bg="base.800" sx={props.sx}>
      <Heading size="sm">{props.title}</Heading>
    </Flex>
  );
});

StickyScrollableHeading.displayName = 'StickyScrollableHeading';

type StickyScrollableContentProps = PropsWithChildren<{ sx?: SystemStyleObject }>;

const StickyScrollableContent = memo((props: StickyScrollableContentProps) => {
  return (
    <Flex p={4} borderRadius="base" bg="base.750" flexDir="column" gap={4} sx={props.sx}>
      {props.children}
    </Flex>
  );
});

StickyScrollableContent.displayName = 'StickyScrollableContent';

type StickyScrollableProps = PropsWithChildren<{
  title: string;
  headingSx?: SystemStyleObject;
  contentSx?: SystemStyleObject;
}>;

export const StickyScrollable = memo((props: StickyScrollableProps) => {
  return (
    <Flex key={props.title} flexDir="column">
      <StickyScrollableHeading title={props.title} sx={props.headingSx} />
      <StickyScrollableContent sx={props.contentSx}>{props.children}</StickyScrollableContent>
    </Flex>
  );
});

StickyScrollable.displayName = 'StickyScrollable';
