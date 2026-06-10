import { Box, HStack, Icon, Menu, Portal, Text, VStack } from '@chakra-ui/react';
import { useEffect, useRef } from 'react';
import { PiCaretDownBold, PiCheckBold, PiLockSimpleFill, PiLightningFill } from 'react-icons/pi';

import { useWorkbench } from '../WorkbenchContext';
import {
  formatRoute,
  invocationSources,
  isInvocationRouteValid,
  resolveInvocationRoute,
  resultDestinations,
} from '../invocation';
import type { InvocationSourceId, ResultDestination } from '../types';

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
const CONTROL_WIDTH = '13.5rem';

export const InvokeControl = () => {
  const { activeProject, dispatch } = useWorkbench();
  const { invocation } = activeProject;
  const resolvedRoute = resolveInvocationRoute(activeProject);
  const isLocked = invocation.sourceLocked || invocation.destinationLocked;
  const isValid = isInvocationRouteValid(resolvedRoute);
  const routeLabel = formatRoute(resolvedRoute);
  const invokeLabel = isValid ? `Invoke — ${routeLabel}` : `Invoke unavailable — ${resolvedRoute.validationMessage}`;
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

  return (
    <HStack flexShrink={0} gap="0" h="7" overflow="hidden" rounded="xs" shadow="sm" w={CONTROL_WIDTH}>
      <HStack
        aria-disabled={!isValid}
        aria-label={invokeLabel}
        as="button"
        bg={isValid ? 'accent.invoke' : 'bg.surface'}
        color={isValid ? 'accent.invokeFg' : 'fg.muted'}
        flex="1"
        gap="2"
        h="full"
        minW="0"
        px="3"
        transition="filter 0.12s ease"
        title={resolvedRoute.validationMessage}
        onClick={() => {
          if (!isValid) {
            return;
          }

          dispatch({
            backendSupportsCancellation: true,
            route: resolvedRoute,
            type: 'submitResolvedInvocationSnapshot',
          });
        }}
        _hover={isValid ? { filter: 'brightness(1.05)' } : undefined}
        _active={{ filter: 'brightness(0.96)' }}
      >
        <Icon as={PiLightningFill} boxSize="4" flexShrink={0} />
        <VStack align="start" gap="0" minW="0">
          <Text fontSize="sm" fontWeight="800" lineHeight="1.1">
            Invoke
          </Text>
          <HStack gap="1" maxW="full">
            <Text fontSize="2xs" fontWeight="600" lineHeight="1.1" opacity="0.85" truncate>
              {routeLabel}
            </Text>
            {isLocked ? <Icon as={PiLockSimpleFill} boxSize="2.5" flexShrink={0} /> : null}
          </HStack>
        </VStack>
      </HStack>

      <Menu.Root positioning={{ placement: 'bottom-start' }}>
        <Menu.Trigger asChild>
          <Box
            aria-label="Open invocation controller"
            as="button"
            bg="accent.invoke"
            borderLeftWidth="1px"
            borderColor="blackAlpha.300"
            color="accent.invokeFg"
            display="grid"
            h="full"
            placeItems="center"
            px="2"
            transition="filter 0.12s ease"
            _hover={{ filter: 'brightness(1.05)' }}
          >
            <Icon as={PiCaretDownBold} boxSize="3" />
          </Box>
        </Menu.Trigger>
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
                      <Icon as={PiCheckBold} boxSize="3" />
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
                      <Icon as={PiCheckBold} boxSize="3" />
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
                <Icon as={PiLockSimpleFill} boxSize="3" opacity={invocation.sourceLocked ? 1 : 0.35} />
                <Menu.ItemText>{invocation.sourceLocked ? 'Unlock source' : 'Lock source'}</Menu.ItemText>
              </Menu.Item>
              <Menu.Item
                value="lock-destination"
                closeOnSelect={false}
                onClick={() => dispatch({ type: 'toggleDestinationLock' })}
              >
                <Icon as={PiLockSimpleFill} boxSize="3" opacity={invocation.destinationLocked ? 1 : 0.35} />
                <Menu.ItemText>
                  {invocation.destinationLocked ? 'Unlock destination' : 'Lock destination'}
                </Menu.ItemText>
              </Menu.Item>
            </Menu.Content>
          </Menu.Positioner>
        </Portal>
      </Menu.Root>
    </HStack>
  );
};
