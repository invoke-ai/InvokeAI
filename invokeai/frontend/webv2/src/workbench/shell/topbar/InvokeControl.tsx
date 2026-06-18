import type { InvocationSourceId, Project, ResultDestination } from '@workbench/types';

import { Flex, Group, HStack, Icon, Menu, Portal, Separator, Stack, Text, VStack } from '@chakra-ui/react';
import { Button, IconButton, Tooltip } from '@workbench/components/ui';
import { sanitizeBatchCount } from '@workbench/generation/batch';
import {
  formatRoute,
  getDestinationLabel,
  invocationSources,
  isInvocationRouteValid,
  resolveInvocationRoute,
  resultDestinations,
} from '@workbench/invocation';
import { ensureModelsLoaded, useModelsSnapshot } from '@workbench/models/modelsStore';
import { getProjectWidgetValues } from '@workbench/widgetState';
import { useActiveProject, useWorkbenchDispatch, useWorkbenchSelector } from '@workbench/WorkbenchContext';
import { useInvocationTemplatesSnapshot } from '@workbench/workflows/templates';
import { CheckIcon, ChevronDownIcon, LockKeyholeIcon, SparklesIcon } from 'lucide-react';
import { useEffect, useRef } from 'react';

/**
 * Fixed-width global Invoke control.
 *
 * The defining Phase 1 requirement: the primary label stays `Invoke`, the
 * secondary line shows the resolved `Source → Destination` route with a compact
 * lock indicator, and the control reserves stable horizontal space so project
 * tabs never shift when the route text changes. The caret opens the Invocation
 * Controller menu (source / destination / lock), which is wired to project-owned
 * state even though full invocation lands in Phase 4.
 */
const CONTROL_WIDTH = '12rem';

const getBatchCount = (values: Record<string, unknown>): number => {
  const batchCount = values.batchCount;

  return sanitizeBatchCount(batchCount);
};

const compactBlockingReason = (reason: string): string => {
  if (reason === 'The project graph has no nodes. Add nodes in the Workflow view.') {
    return 'No nodes in graph';
  }

  return reason.replace(/^The /, '').replace(/project graph/i, 'workflow');
};

const InvokeTooltipContent = ({
  blockingReasons,
  isValid,
  project,
}: {
  blockingReasons: string[];
  isValid: boolean;
  project: Project;
}) => {
  const batchCount = getBatchCount(getProjectWidgetValues(project, 'generate'));
  const destination = getDestinationLabel(project.invocation.destination);
  const summary =
    project.invocation.sourceId === 'generate'
      ? `1 prompt × ${batchCount} iteration${batchCount === 1 ? '' : 's'} → ${batchCount} generation${batchCount === 1 ? '' : 's'}`
      : `Workflow × ${batchCount} run${batchCount === 1 ? '' : 's'} → ${batchCount} generation${batchCount === 1 ? '' : 's'}`;

  return (
    <Stack gap="1.5" minW="14rem" p="2">
      <Text fontSize="xs" fontWeight="800">
        {isValid ? 'Add to Queue' : 'Unable to Queue'}
      </Text>
      <Text color="fg.muted" fontSize="xs">
        {summary}
      </Text>
      {blockingReasons.length > 0 ? (
        <>
          <Separator borderColor="border.subtle" />
          <Stack gap="1">
            {blockingReasons.map((reason) => (
              <HStack key={reason} align="start" gap="1.5">
                <Text color="fg.subtle" fontSize="xs" lineHeight="1.35">
                  •
                </Text>
                <Text color="fg.muted" fontSize="xs" lineHeight="1.35">
                  {compactBlockingReason(reason)}
                </Text>
              </HStack>
            ))}
          </Stack>
        </>
      ) : (
        <>
          <Separator borderColor="border.subtle" />
          <Text color="fg.muted" fontSize="xs">
            Adding images to {destination}
          </Text>
        </>
      )}
    </Stack>
  );
};

export const InvokeControl = () => {
  const activeProject = useActiveProject();
  const dispatch = useWorkbenchDispatch();
  const backendConnectionStatus = useWorkbenchSelector((snapshot) => snapshot.state.backendConnection.status);
  const { models, status: modelsStatus } = useModelsSnapshot();
  const availabilityModels = modelsStatus === 'loaded' ? models : undefined;
  const { invocation } = activeProject;

  // Project-graph route validation reads the invocation templates imperatively;
  // subscribing here keeps the resolved route live while they load.
  useInvocationTemplatesSnapshot();

  const resolvedRoute = resolveInvocationRoute(activeProject, 'global', activeProject.invocation, availabilityModels);
  const isLocked = invocation.sourceLocked || invocation.destinationLocked;
  const isConnected = backendConnectionStatus === 'connected';
  // Legacy parity: the queue button disables with the full list of reasons,
  // including a disconnected backend.
  const blockingReasons = [
    ...(isConnected ? [] : ['The backend is disconnected.']),
    ...resolvedRoute.validationReasons,
  ];
  const isValid = isInvocationRouteValid(resolvedRoute) && isConnected;
  const routeLabel = isValid ? formatRoute(resolvedRoute) : (blockingReasons[0] ?? formatRoute(resolvedRoute));
  const resolvedRouteRef = useRef(resolvedRoute);
  const isValidRef = useRef(isValid);
  const modelsRef = useRef(availabilityModels);

  resolvedRouteRef.current = resolvedRoute;
  isValidRef.current = isValid;
  modelsRef.current = availabilityModels;

  useEffect(() => {
    ensureModelsLoaded();
  }, []);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (!(event.metaKey || event.ctrlKey) || event.key !== 'Enter') {
        return;
      }

      event.preventDefault();

      if (!isValidRef.current) {
        return;
      }

      dispatch({
        backendSupportsCancellation: true,
        models: modelsRef.current,
        route: resolvedRouteRef.current,
        type: 'submitResolvedInvocationSnapshot',
      });
    };

    window.addEventListener('keydown', onKeyDown);

    return () => {
      window.removeEventListener('keydown', onKeyDown);
    };
  }, [dispatch]);

  const onInvoke = () => {
    if (!isValid) {
      return;
    }

    dispatch({
      backendSupportsCancellation: true,
      models: availabilityModels,
      route: resolvedRouteRef.current,
      type: 'submitResolvedInvocationSnapshot',
    });
  };

  return (
    <Flex>
      <Menu.Root positioning={{ placement: 'bottom-end' }}>
        <Group attached>
          <Tooltip
            content={
              <InvokeTooltipContent blockingReasons={blockingReasons} isValid={isValid} project={activeProject} />
            }
            contentProps={{ p: '0' }}
            openDelay={200}
            showArrow
          >
            {/* aria-disabled (not disabled) keeps hover alive so the reasons tooltip can show. */}
            <Button
              aria-disabled={!isValid}
              colorPalette="brand"
              cursor={isValid ? undefined : 'not-allowed'}
              opacity={isValid ? undefined : 0.6}
              size="sm"
              roundedEnd="none"
              onClick={onInvoke}
              w={CONTROL_WIDTH}
              minW="0"
              justifyContent="start"
            >
              <Icon as={SparklesIcon} boxSize="4" flexShrink={0} />
              <VStack align="start" gap="0" minW="0">
                <Text fontSize="sm" fontWeight="700" lineHeight="1">
                  Invoke
                </Text>
                <HStack gap="1" maxW="full">
                  <Text fontSize="2xs" fontWeight="600" lineHeight="1.1" opacity="0.85" truncate>
                    {routeLabel}
                  </Text>
                  {isLocked ? <Icon as={LockKeyholeIcon} boxSize="2.5" flexShrink={0} /> : null}
                </HStack>
              </VStack>
            </Button>
          </Tooltip>
          <Menu.Trigger asChild>
            <IconButton colorPalette="brand" size="sm" minW="0" w="7">
              <ChevronDownIcon />
            </IconButton>
          </Menu.Trigger>
        </Group>
        <Portal>
          <Menu.Positioner>
            <Menu.Content minW="14rem">
              <Menu.RadioItemGroup
                value={invocation.sourceId}
                onValueChange={(event) =>
                  dispatch({ sourceId: event.value as InvocationSourceId, type: 'setInvocationSource' })
                }
              >
                <Menu.ItemGroupLabel color="fg.subtle" fontSize="2xs" textTransform="uppercase">
                  Source
                </Menu.ItemGroupLabel>
                {invocationSources.map((source) => (
                  <Menu.RadioItem
                    key={source.id}
                    value={source.id}
                    disabled={!source.available}
                    _disabled={{ opacity: 0.4 }}
                  >
                    <Menu.ItemText>{source.label}</Menu.ItemText>
                    {source.available ? null : (
                      <Text color="fg.subtle" fontSize="2xs" ms="auto">
                        Soon
                      </Text>
                    )}
                    <Menu.ItemIndicator>
                      <Icon as={CheckIcon} boxSize="3" />
                    </Menu.ItemIndicator>
                  </Menu.RadioItem>
                ))}
              </Menu.RadioItemGroup>

              <Menu.Separator borderColor="border.subtle" />

              <Menu.RadioItemGroup
                value={invocation.destination}
                onValueChange={(event) =>
                  dispatch({ destination: event.value as ResultDestination, type: 'setInvocationDestination' })
                }
              >
                <Menu.ItemGroupLabel color="fg.subtle" fontSize="2xs" textTransform="uppercase">
                  Destination
                </Menu.ItemGroupLabel>
                {resultDestinations.map((destination) => (
                  <Menu.RadioItem key={destination.id} value={destination.id}>
                    <Menu.ItemText>{destination.label}</Menu.ItemText>
                    <Menu.ItemIndicator>
                      <Icon as={CheckIcon} boxSize="3" />
                    </Menu.ItemIndicator>
                  </Menu.RadioItem>
                ))}
              </Menu.RadioItemGroup>

              <Menu.Separator borderColor="border.subtle" />

              <Menu.Item
                value="lock-source"
                closeOnSelect={false}
                onClick={() => dispatch({ type: 'toggleSourceLock' })}
              >
                <Icon as={LockKeyholeIcon} boxSize="3" opacity={invocation.sourceLocked ? 1 : 0.35} />
                <Menu.ItemText>{invocation.sourceLocked ? 'Unlock source' : 'Lock source'}</Menu.ItemText>
              </Menu.Item>
              <Menu.Item
                value="lock-destination"
                closeOnSelect={false}
                onClick={() => dispatch({ type: 'toggleDestinationLock' })}
              >
                <Icon as={LockKeyholeIcon} boxSize="3" opacity={invocation.destinationLocked ? 1 : 0.35} />
                <Menu.ItemText>
                  {invocation.destinationLocked ? 'Unlock destination' : 'Lock destination'}
                </Menu.ItemText>
              </Menu.Item>
            </Menu.Content>
          </Menu.Positioner>
        </Portal>
      </Menu.Root>
    </Flex>
  );
};
