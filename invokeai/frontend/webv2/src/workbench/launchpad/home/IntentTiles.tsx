import type { LaunchpadStartEntry } from '@workbench/launchpad/startEntries';

import { Icon, SimpleGrid, Stack, Text } from '@chakra-ui/react';
import { Link } from '@tanstack/react-router';
import { LAUNCHPAD_START_ENTRIES } from '@workbench/launchpad/startEntries';
import { useTranslation } from 'react-i18next';

const TILE_COLUMNS = { base: 1, lg: 5, sm: 2 } as const;
// Tiles follow the theme's control cursor convention.
const TILE_LINK_STYLE = { cursor: 'default' } as const;
const TILE_HOVER = { bg: 'bg.muted', borderColor: 'border.emphasized' } as const;
const TILE_TRANSITION =
  'border-color var(--wb-motion-duration-medium) ease, background var(--wb-motion-duration-medium) ease';

const IntentTile = ({ entry }: { entry: LaunchpadStartEntry }) => {
  const { t } = useTranslation();

  return (
    <Link search={entry.search} style={TILE_LINK_STYLE} to="/app">
      <Stack
        bg="bg.subtle"
        borderColor="border.subtle"
        borderWidth="1px"
        gap="1"
        h="full"
        p="3"
        rounded="lg"
        transition={TILE_TRANSITION}
        _hover={TILE_HOVER}
      >
        <Icon as={entry.icon} boxSize="4" color="fg.muted" />
        <Text fontSize="xs" fontWeight="600">
          {t(entry.labelKey)}
        </Text>
        <Text color="fg.muted" fontSize="2xs">
          {t(entry.descriptionKey)}
        </Text>
      </Stack>
    </Link>
  );
};

export const IntentTiles = () => (
  <SimpleGrid columns={TILE_COLUMNS} gap="3">
    {LAUNCHPAD_START_ENTRIES.map((entry) => (
      <IntentTile entry={entry} key={entry.id} />
    ))}
  </SimpleGrid>
);
