import type { ReactNode } from 'react';

import { Badge, HStack, Icon, Spinner, Stack, Text } from '@chakra-ui/react';
import { Button, Panel } from '@workbench/components/ui';
import { useActiveInstallSources } from '@workbench/models/installsStore';
import { openModelsCenterTab } from '@workbench/models/uiStore';
import { DownloadIcon } from 'lucide-react';

/**
 * The shared row used by every installable-source list (starter models,
 * HuggingFace files, folder-scan results, related models): title + badges on
 * top, optional description below, action on the right.
 */
export const SourceListItem = ({
  badges,
  description,
  title,
  titleTooltip,
  trailing,
}: {
  badges?: ReactNode;
  description?: ReactNode;
  title: string;
  titleTooltip?: string;
  trailing?: ReactNode;
}) => (
  <Panel alignItems="center" flexDirection="row" gap="3" p="2.5">
    <Stack flex="1" gap="0.5" minW="0">
      <HStack gap="1.5" minW="0">
        <Text fontSize="xs" fontWeight="600" title={titleTooltip} truncate>
          {title}
        </Text>
        {badges}
      </HStack>
      {description ? (
        <Text color="fg.subtle" fontSize="2xs" lineClamp={2}>
          {description}
        </Text>
      ) : null}
    </Stack>
    {trailing}
  </Panel>
);

/**
 * Install action with live state: idle → Install button; queuing/active →
 * disabled "Installing…" with a jump to the install queue; done → badge.
 */
export const InstallSourceButton = ({
  isInstalled = false,
  isPending = false,
  onInstall,
  source,
}: {
  /** Already in the library (renders a static badge). */
  isInstalled?: boolean;
  /** The install POST is in flight (before a job exists). */
  isPending?: boolean;
  onInstall: () => void;
  /** Source string used to match an active install job. */
  source: string;
}) => {
  const activeSources = useActiveInstallSources();
  const isInstalling = isPending || activeSources.has(source);

  if (isInstalled) {
    return (
      <Badge colorPalette="green" flexShrink={0} fontSize="2xs" size="sm" variant="surface">
        Installed
      </Badge>
    );
  }

  if (isInstalling) {
    return (
      <HStack flexShrink={0} gap="1.5">
        <Badge colorPalette="blue" fontSize="2xs" size="sm" variant="surface">
          <Spinner borderWidth="1.5px" boxSize="2.5" />
          Installing
        </Badge>
        <Button size="2xs" variant="ghost" onClick={() => openModelsCenterTab('queue')}>
          View queue
        </Button>
      </HStack>
    );
  }

  return (
    <Button flexShrink={0} size="2xs" variant="outline" onClick={onInstall}>
      <Icon as={DownloadIcon} boxSize="3" />
      Install
    </Button>
  );
};
