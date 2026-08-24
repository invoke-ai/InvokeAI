import type { LaunchpadIntentId } from '@workbench/launchpad/intents';
import type { LucideIcon } from 'lucide-react';

import { Icon, SimpleGrid, Stack, Text } from '@chakra-ui/react';
import { Link } from '@tanstack/react-router';
import { LAUNCHPAD_INTENT_IDS } from '@workbench/launchpad/intents';
import { BlocksIcon, BrushIcon, ClapperboardIcon, ScalingIcon, TypeIcon } from 'lucide-react';
import { useMemo } from 'react';
import { useTranslation } from 'react-i18next';

/**
 * "What do you want to make?", asked before "which file do you want to open?".
 *
 * Each tile opens a fresh draft already arranged for that kind of work. The
 * old frontend put this inside the editor as a per-tab panel; at the home
 * level it is the first move rather than a thing you find after arriving.
 */

const TILE_COLUMNS = { base: 1, lg: 5, sm: 2 } as const;
const TILE_HOVER = { bg: 'bg.muted', borderColor: 'border.emphasized' } as const;
const TILE_TRANSITION =
  'border-color var(--wb-motion-duration-medium) ease, background var(--wb-motion-duration-medium) ease';

const INTENT_ICON: Record<LaunchpadIntentId, LucideIcon> = {
  canvas: BrushIcon,
  generate: TypeIcon,
  upscale: ScalingIcon,
  video: ClapperboardIcon,
  workflow: BlocksIcon,
};

// Spelled out rather than interpolated, so the translation-key test can see
// them and fail the build if a string goes missing.
const INTENT_LABEL_KEY: Record<LaunchpadIntentId, string> = {
  canvas: 'launchpad.home.intents.canvas',
  generate: 'launchpad.home.intents.generate',
  upscale: 'launchpad.home.intents.upscale',
  video: 'launchpad.home.intents.video',
  workflow: 'launchpad.home.intents.workflow',
};

const INTENT_DESCRIPTION_KEY: Record<LaunchpadIntentId, string> = {
  canvas: 'launchpad.home.intents.canvasDescription',
  generate: 'launchpad.home.intents.generateDescription',
  upscale: 'launchpad.home.intents.upscaleDescription',
  video: 'launchpad.home.intents.videoDescription',
  workflow: 'launchpad.home.intents.workflowDescription',
};

const IntentTile = ({ id }: { id: LaunchpadIntentId }) => {
  const { t } = useTranslation();
  const search = useMemo(() => ({ intent: id, new: true }) as const, [id]);

  return (
    <Link search={search} to="/app">
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
        <Icon as={INTENT_ICON[id]} boxSize="4" color="fg.muted" />
        <Text fontSize="xs" fontWeight="600">
          {t(INTENT_LABEL_KEY[id])}
        </Text>
        <Text color="fg.muted" fontSize="2xs">
          {t(INTENT_DESCRIPTION_KEY[id])}
        </Text>
      </Stack>
    </Link>
  );
};

export const IntentTiles = () => (
  <SimpleGrid columns={TILE_COLUMNS} gap="3">
    {LAUNCHPAD_INTENT_IDS.map((id) => (
      <IntentTile id={id} key={id} />
    ))}
  </SimpleGrid>
);
