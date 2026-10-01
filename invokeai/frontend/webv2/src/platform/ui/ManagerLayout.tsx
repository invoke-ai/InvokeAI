import type { ReactNode } from 'react';

import { Box, Flex, HStack, Text } from '@chakra-ui/react';

/**
 * The full-bleed list/detail shell shared by the Launchpad managers (models, nodes, fonts, preferences); the
 * column and the detail header share one header height so their bottom borders line up.
 */

const COLUMN_WIDTH = 'clamp(22rem, 32vw, 28rem)';
const HEADER_MIN_HEIGHT = '2.75rem';

export const ManagerColumn = ({
  actions,
  children,
  count,
  title,
}: {
  /** Controls at the end of the column header. */
  actions?: ReactNode;
  children: ReactNode;
  count?: ReactNode;
  title: string;
}) => (
  <Flex borderEndWidth="1px" direction="column" flexShrink={0} h="full" minH="0" position="relative" w={COLUMN_WIDTH}>
    <HStack borderBottomWidth="1px" flexShrink={0} gap="2" minH={HEADER_MIN_HEIGHT} px="3">
      <Text as="h2" fontSize="sm" fontWeight="700">
        {title}
      </Text>
      {count === undefined ? null : (
        <Text color="fg.muted" fontSize="xs" fontVariantNumeric="tabular-nums">
          {count}
        </Text>
      )}
      {actions ? <Box ms="auto">{actions}</Box> : null}
    </HStack>
    {children}
  </Flex>
);

/** Holds the detail pane's tabs or title, bottom-aligned so a tab list's indicator sits on the border. */
export const ManagerDetailHeader = ({ children }: { children: ReactNode }) => (
  <Flex align="flex-end" borderBottomWidth="1px" flexShrink={0} minH={HEADER_MIN_HEIGHT} px="2">
    {children}
  </Flex>
);
