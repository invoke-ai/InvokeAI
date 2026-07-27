import type { InvocationRoute, InvocationSourceId, ResultDestination } from '@workbench/invocationContracts';

import { Flex, Group, HStack, Icon, Menu, Portal, Separator, Stack, Text, VStack } from '@chakra-ui/react';
import { flushGenerateDrafts, useDynamicPrompts, type DynamicPromptsExpansion } from '@features/generation/react';
import { sanitizeBatchCount, sanitizeDynamicPromptsConfig } from '@features/generation/settings';
import { ensureModelsLoaded, useModelsSelector } from '@features/models';
import { useInvocationTemplatesSelector } from '@features/workflow/react';
import { useMountEffect } from '@platform/react/useMountEffect';
import {
  assertAccountScopeCurrent,
  captureAccountScope,
  isAccountScopeCurrent,
} from '@platform/state/accountLifecycle';
import { Button, IconButton, Tooltip } from '@platform/ui';
import {
  formatRoute,
  createInvocationRouteInputSelector,
  getDestinationLabel,
  invocationSources,
  isInvocationRouteValid,
  resolveInvocationRoute,
  resolveInvocationRouteInput,
  resultDestinations,
} from '@workbench/invocation';
import { submitResolvedInvocation } from '@workbench/invocationSubmit';
import {
  useActiveProjectSelector,
  useWorkbenchCommands,
  useWorkbenchQueries,
  useWorkbenchSelector,
} from '@workbench/WorkbenchContext';
import { CheckIcon, ChevronDownIcon, LockKeyholeIcon, SparklesIcon } from 'lucide-react';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

const CONTROL_WIDTH = '10rem';
const selectInvocationRouteInput = createInvocationRouteInputSelector();
const MENU_POSITIONING = { placement: 'bottom-end' } as const;
const TOOLTIP_CONTENT_PROPS = { p: '0' };
const DISABLED_PROPS = { opacity: 0.4 };

/** Reads the persisted dynamic prompt fields off untyped widget values. */
const readDynamicPromptsConfig = (values: Record<string, unknown>) => ({
  combinatorial: values.dynamicPromptsCombinatorial,
  maxPrompts: values.dynamicPromptsMaxPrompts,
  sampleSeed: values.dynamicPromptsSampleSeed,
  seedBehaviour: values.dynamicPromptsSeedBehaviour,
});

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

const plural = (count: number, noun: string): string => `${count} ${noun}${count === 1 ? '' : 's'}`;

const InvokeTooltipContent = ({
  blockingReasons,
  sourceValues,
  invocation,
  isValid,
  promptExpansion,
}: {
  blockingReasons: string[];
  sourceValues: Record<string, unknown>;
  invocation: InvocationRoute;
  isValid: boolean;
  promptExpansion: DynamicPromptsExpansion;
}) => {
  const batchCount = getBatchCount(sourceValues);
  const destination = getDestinationLabel(invocation.destination);
  const promptCount = promptExpansion.count;
  const summary =
    invocation.sourceId === 'generate' || invocation.sourceId === 'upscale'
      ? promptExpansion.isLoading
        ? 'Expanding prompts…'
        : `${plural(promptCount, 'prompt')} × ${plural(batchCount, 'iteration')} → ${plural(promptCount * batchCount, 'generation')}`
      : `Workflow × ${plural(batchCount, 'run')} → ${plural(batchCount, 'generation')}`;

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
  const { t } = useTranslation();
  const routeInput = useActiveProjectSelector(selectInvocationRouteInput);
  const commands = useWorkbenchCommands();
  const { generation } = commands;
  const queries = useWorkbenchQueries();
  const backendConnectionStatus = useWorkbenchSelector((snapshot) => snapshot.backendConnection.status);
  const models = useModelsSelector((snapshot) => snapshot.models);
  const modelsStatus = useModelsSelector((snapshot) => snapshot.status);
  const availabilityModels = modelsStatus === 'loaded' ? models : undefined;
  const { invocation } = routeInput;

  // Project-graph route validation reads the invocation templates imperatively;
  // subscribing here keeps the resolved route live while they load.
  useInvocationTemplatesSelector((snapshot) => snapshot.status);

  const resolvedRoute = resolveInvocationRouteInput(routeInput, 'global', routeInput.invocation, availabilityModels);
  const isLocked = invocation.sourceLocked || invocation.destinationLocked;
  const isConnected = backendConnectionStatus === 'connected';

  // Observing the shared expansion cache here keeps the count honest and costs
  // nothing extra: the Generate preview and the submit path use the same key.
  const promptExpansion = useDynamicPrompts(
    typeof routeInput.generateValues.positivePrompt === 'string' ? routeInput.generateValues.positivePrompt : '',
    invocation.sourceId === 'generate' || invocation.sourceId === 'canvas'
      ? sanitizeDynamicPromptsConfig(readDynamicPromptsConfig(routeInput.generateValues))
      : null
  );
  const blockingReasons = useMemo(
    () => [
      ...(isConnected ? [] : ['The backend is disconnected.']),
      ...(promptExpansion.isError ? ['The prompt could not be expanded.'] : []),
      ...resolvedRoute.validationReasons,
    ],
    [isConnected, promptExpansion.isError, resolvedRoute.validationReasons]
  );
  const isValid = isInvocationRouteValid(resolvedRoute) && isConnected;
  const routeLabel = isValid ? formatRoute(resolvedRoute) : (blockingReasons[0] ?? formatRoute(resolvedRoute));
  useMountEffect(() => {
    void ensureModelsLoaded();
  });

  const onInvoke = useCallback(async () => {
    const owner = captureAccountScope();
    flushGenerateDrafts();

    try {
      const { prepareCanvasInvocation } = await import('@workbench/widgets/canvas/invoke/prepareCanvasInvocation');

      assertAccountScopeCurrent(owner);
      const snapshot = queries.getSnapshot();
      const postFlushRoute = resolveInvocationRoute(
        snapshot.activeProject,
        'global',
        snapshot.activeProject.invocation,
        availabilityModels
      );

      if (!isInvocationRouteValid(postFlushRoute) || snapshot.backendConnection.status !== 'connected') {
        return;
      }

      submitResolvedInvocation({
        commands,
        formatControlLayerError: (code, layerName) =>
          t('widgets.layers.control.invalidLayer', {
            name: layerName,
            reason: t(`widgets.layers.control.validation.${code}`),
          }),
        models: availabilityModels,
        owner,
        prepareCanvasInvocation,
        project: snapshot.activeProject,
        route: postFlushRoute,
      });
    } catch (error) {
      if (!isAccountScopeCurrent(owner)) {
        return;
      }

      throw error;
    }
  }, [availabilityModels, commands, queries, t]);
  const tooltipContent = useMemo(
    () => (
      <InvokeTooltipContent
        blockingReasons={blockingReasons}
        sourceValues={
          routeInput.invocation.sourceId === 'upscale' ? routeInput.upscaleValues : routeInput.generateValues
        }
        invocation={invocation}
        isValid={isValid}
        promptExpansion={promptExpansion}
      />
    ),
    [
      blockingReasons,
      invocation,
      isValid,
      promptExpansion,
      routeInput.generateValues,
      routeInput.invocation.sourceId,
      routeInput.upscaleValues,
    ]
  );
  const handleSourceChange = useCallback(
    (event: { value: string }) => generation.setSource(event.value as InvocationSourceId),
    [generation]
  );
  const handleDestinationChange = useCallback(
    (event: { value: string }) => generation.setDestination(event.value as ResultDestination),
    [generation]
  );
  return (
    <Flex>
      <Menu.Root positioning={MENU_POSITIONING}>
        <Group attached>
          <Tooltip content={tooltipContent} contentProps={TOOLTIP_CONTENT_PROPS} openDelay={200} showArrow>
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
                <HStack gap="1" maxW="full" mb="-0.5">
                  <Text fontSize="0.58rem" lineHeight="1.1" opacity="0.85" truncate>
                    {routeLabel}
                  </Text>
                  {isLocked ? <Icon as={LockKeyholeIcon} boxSize="2.5" flexShrink={0} /> : null}
                </HStack>
              </VStack>
            </Button>
          </Tooltip>
          <Menu.Trigger asChild>
            <IconButton
              aria-label="Choose invocation source and destination"
              colorPalette="brand"
              minW="0"
              size="sm"
              w="7"
            >
              <ChevronDownIcon />
            </IconButton>
          </Menu.Trigger>
        </Group>
        <Portal>
          <Menu.Positioner>
            <Menu.Content minW="14rem">
              <Menu.RadioItemGroup value={invocation.sourceId} onValueChange={handleSourceChange}>
                <Menu.ItemGroupLabel color="fg.subtle" fontSize="2xs" textTransform="uppercase">
                  Source
                </Menu.ItemGroupLabel>
                {invocationSources.map((source) => (
                  <Menu.RadioItem
                    key={source.id}
                    value={source.id}
                    disabled={!source.available}
                    _disabled={DISABLED_PROPS}
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

              <Menu.RadioItemGroup value={invocation.destination} onValueChange={handleDestinationChange}>
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

              <Menu.Item value="lock-source" closeOnSelect={false} onClick={generation.toggleSourceLock}>
                <Icon as={LockKeyholeIcon} boxSize="3" opacity={invocation.sourceLocked ? 1 : 0.35} />
                <Menu.ItemText>{invocation.sourceLocked ? 'Unlock source' : 'Lock source'}</Menu.ItemText>
              </Menu.Item>
              <Menu.Item value="lock-destination" closeOnSelect={false} onClick={generation.toggleDestinationLock}>
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
