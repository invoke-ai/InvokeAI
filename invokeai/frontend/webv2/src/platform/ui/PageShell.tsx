import type { SystemStyleObject } from '@chakra-ui/react';
import type { ReactNode } from 'react';

import { Flex, HStack, Heading, Stack, Text } from '@chakra-ui/react';

import { Scrollable } from './Scrollable';

/**
 * Use scroll=page for whole-page scrolling; content pins the header and lets virtualized children own their scroll
 * element. Full-bleed views provide their own shell.
 */

const MEASURE_SX: SystemStyleObject = {
  maxW: '6xl',
  mx: 'auto',
  px: { base: 4, md: 8 },
  w: 'full',
};

const PAGE_SCROLL_MEASURE_SX: SystemStyleObject = { ...MEASURE_SX, py: { base: 4, md: 8 } };
const CONTENT_SCROLL_HEADER_SX: SystemStyleObject = { ...MEASURE_SX, pb: 4, pt: { base: 4, md: 8 } };

export interface PageShellProps {
  /** The section's visible heading. */
  title: string;
  /** Override the scroll region's accessible name; ignored for scroll=content, where children own the region. */
  regionLabel?: string;
  /** One line under the heading; omit when the section speaks for itself. */
  description?: string;
  /** Controls aligned opposite the heading. */
  actions?: ReactNode;
  /** Rendered between the header and the content — alerts, banners, toolbars. */
  banner?: ReactNode;
  scroll?: 'page' | 'content';
  children: ReactNode;
}

const PageShellHeader = ({
  actions,
  description,
  title,
}: Pick<PageShellProps, 'actions' | 'description' | 'title'>) => (
  <Flex align="center" gap="3" justify="space-between" wrap="wrap">
    <Stack gap="0.5">
      <Heading fontSize="xl" fontWeight="700">
        {title}
      </Heading>
      {description ? (
        <Text color="fg.muted" fontSize="xs">
          {description}
        </Text>
      ) : null}
    </Stack>
    {actions ? (
      <HStack gap="2" wrap="wrap">
        {actions}
      </HStack>
    ) : null}
  </Flex>
);

export const PageShell = ({
  actions,
  banner,
  children,
  description,
  regionLabel,
  scroll = 'page',
  title,
}: PageShellProps) => {
  if (scroll === 'content') {
    return (
      <Flex direction="column" h="full" minH="0" w="full">
        <Stack css={CONTENT_SCROLL_HEADER_SX} flexShrink={0} gap="4">
          <PageShellHeader actions={actions} description={description} title={title} />
          {banner}
        </Stack>
        <Flex direction="column" flex="1" minH="0" w="full">
          {children}
        </Flex>
      </Flex>
    );
  }

  return (
    <Scrollable h="full" label={regionLabel ?? title} minH="0">
      <Stack css={PAGE_SCROLL_MEASURE_SX} gap="5">
        <PageShellHeader actions={actions} description={description} title={title} />
        {banner}
        {children}
      </Stack>
    </Scrollable>
  );
};
