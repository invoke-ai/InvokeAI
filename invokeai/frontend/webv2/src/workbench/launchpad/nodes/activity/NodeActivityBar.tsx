/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import { Badge, Box, Collapsible, Flex, HStack, Icon, Spinner, Text } from '@chakra-ui/react';
import { Button } from '@workbench/components/ui';
import { clearCustomNodeInstallLog, useCustomNodeInstallLog } from '@workbench/customNodes/installLogStore';
import { setNodeActivityExpanded, useNodesUiSelector } from '@workbench/customNodes/nodesUiStore';
import { ChevronUpIcon, ListOrderedIcon, Trash2Icon } from 'lucide-react';

import { NodeInstallLog } from './NodeInstallLog';

/** Persistent, collapsible install activity footer for the nodes manager detail pane. */
export const NodeActivityBar = () => {
  const activityExpanded = useNodesUiSelector((snapshot) => snapshot.activityExpanded);
  const log = useCustomNodeInstallLog();
  const installingCount = log.filter((entry) => entry.status === 'installing').length;
  const summary =
    installingCount > 0
      ? `${log.find((entry) => entry.status === 'installing')?.name ?? 'Installing'}${
          installingCount > 1 ? ` +${installingCount - 1} more` : ''
        }`
      : log.length > 0
        ? `${log.length} recent activit${log.length === 1 ? 'y' : 'ies'}`
        : 'No recent activity';

  return (
    <Collapsible.Root
      bg="bg.subtle"
      borderTopWidth={1}
      flexShrink={0}
      open={activityExpanded}
      overflow="hidden"
      onOpenChange={(event) => setNodeActivityExpanded(event.open)}
    >
      <Collapsible.Content>
        <Flex direction="column" h="min(22rem, 45dvh)" minH="0" overflow="hidden">
          <HStack borderBottomWidth={1} gap="2" justify="space-between" px="3" py="1.5">
            <Text color="fg.subtle" fontSize="2xs" fontWeight="700" textTransform="uppercase">
              Install Activity
            </Text>
            <Button disabled={log.length === 0} size="2xs" variant="ghost" onClick={clearCustomNodeInstallLog}>
              <Icon as={Trash2Icon} boxSize="3" />
              Clear
            </Button>
          </HStack>
          <Box flex="1" minH="0" overflow="hidden" p="2">
            <NodeInstallLog />
          </Box>
        </Flex>
      </Collapsible.Content>

      <HStack gap="1" px="3" py="2">
        <Collapsible.Trigger
          alignItems="center"
          bg="transparent"
          color="inherit"
          display="flex"
          flex="1"
          gap="2"
          minW="0"
          textAlign="start"
          _hover={{ color: 'fg' }}
        >
          {installingCount > 0 ? (
            <Spinner borderWidth="1.5px" boxSize="3.5" color="accent.solid" flexShrink={0} />
          ) : (
            <Icon as={ListOrderedIcon} boxSize="3.5" color="fg.subtle" flexShrink={0} />
          )}
          <Text flex="1" fontSize="xs" fontWeight="600" minW="0" truncate>
            {summary}
          </Text>
          {installingCount > 0 ? (
            <Badge colorPalette="accent" flexShrink={0} fontSize="2xs" size="sm" variant="solid">
              {installingCount}
            </Badge>
          ) : null}
          <Collapsible.Indicator
            _open={{ transform: 'rotate(180deg)' }}
            transition="transform var(--wb-motion-duration-slow)"
          >
            <Icon as={ChevronUpIcon} boxSize="4" color="fg.subtle" flexShrink={0} />
          </Collapsible.Indicator>
        </Collapsible.Trigger>
      </HStack>
    </Collapsible.Root>
  );
};
/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
