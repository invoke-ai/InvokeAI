import { Flex, Group, HStack, Icon, Menu, Portal, Text, VStack } from '@chakra-ui/react';
import { useEffect, useRef } from 'react';
import { CheckIcon, ChevronDownIcon, LockKeyholeIcon, SparklesIcon } from 'lucide-react';

import { useWorkbench } from '../WorkbenchContext';
import {
  formatRoute,
  invocationSources,
  isInvocationRouteValid,
  resolveInvocationRoute,
  resultDestinations,
} from '../invocation';
import type { InvocationSourceId, ResultDestination } from '../types';
import { Button, IconButton } from './ui/Button';

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

export const InvokeControl = () => {
  const { activeProject, dispatch } = useWorkbench();
  const { invocation } = activeProject;
  const resolvedRoute = resolveInvocationRoute(activeProject);
  const isLocked = invocation.sourceLocked || invocation.destinationLocked;
  const isValid = isInvocationRouteValid(resolvedRoute);
  const routeLabel = formatRoute(resolvedRoute);
  const resolvedRouteRef = useRef(resolvedRoute);
  const isValidRef = useRef(isValid);

  resolvedRouteRef.current = resolvedRoute;
  isValidRef.current = isValid;

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
      route: resolvedRouteRef.current,
      type: 'submitResolvedInvocationSnapshot',
    });
  };

  return (
    <Flex>
      <Menu.Root positioning={{ placement: 'bottom-start' }}>
        <Group attached>
          <Button size="sm" onClick={onInvoke} w={CONTROL_WIDTH} minW="0" justifyContent="start">
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
          <Menu.Trigger asChild>
            <IconButton size="sm" minW="0" w="7">
              <ChevronDownIcon />
            </IconButton>
          </Menu.Trigger>
        </Group>
        <Portal>
          <Menu.Positioner>
            <Menu.Content
              bg="bg.surfaceRaised"
              borderWidth="1px"
              borderColor="border.emphasis"
              color="fg.default"
              minW="15rem"
              rounded="lg"
              shadow="lg"
            >
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
