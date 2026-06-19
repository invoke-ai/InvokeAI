import type { LucideIcon } from 'lucide-react';

import { Box, Icon, Stack, Text } from '@chakra-ui/react';
import { Tabs } from '@workbench/components/ui';

import { ResourceLinks } from './ResourceLinks';

export interface LaunchpadNavItem {
  id: string;
  label: string;
  icon: LucideIcon;
}

/**
 * The Launchpad's persistent left rail: top-level section switcher up top,
 * docs/community links pinned to the bottom. Kept deliberately narrow so the
 * page area to its right gets the room — the model manager in particular wants
 * the full width. Renders inside the section `Tabs.Root`, so the triggers carry
 * the roving-focus and selection semantics for free.
 */
export const LaunchpadNav = ({ items }: { items: LaunchpadNavItem[] }) => (
  <Stack
    aria-label="Launchpad sections"
    as="nav"
    borderColor="border.subtle"
    borderEndWidth={{ md: '1px' }}
    flexShrink={0}
    gap="4"
    h="full"
    minH="0"
    p="2"
    w={{ base: 'full', md: '56' }}
  >
    {items.length > 1 ? (
      <Tabs.List flexShrink={0}>
        {items.map((item) => (
          <Tabs.Trigger key={item.id} value={item.id}>
            <Icon as={item.icon} boxSize="3.5" flexShrink={0} />
            <Text truncate>{item.label}</Text>
          </Tabs.Trigger>
        ))}
      </Tabs.List>
    ) : null}

    <Box mt={{ md: 'auto' }}>
      <ResourceLinks />
    </Box>
  </Stack>
);
