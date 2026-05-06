import { Badge, Box, Button, Flex, IconButton, Text, Tooltip } from '@invoke-ai/ui-library';
import type { MouseEvent } from 'react';
import { memo, useCallback } from 'react';
import {
  PiDiceFiveBold,
  PiPushPinSimpleBold,
  PiRepeatBold,
  PiSelectionBold,
  PiSquaresFourBold,
  PiXBold,
} from 'react-icons/pi';

import {
  getWeightBehaviorLabel,
  getWildcardBehaviorLabel,
  type PromptRange,
  type PromptWeightOccurrence,
  type PromptWildcardOccurrence,
  type PromptWorkbenchOccurrence,
} from './occurrences';

type PromptInspectorProps = {
  occurrences: PromptWorkbenchOccurrence[];
  randomRefreshMode: 'manual' | 'per_enqueue';
  fixedWildcardOccurrenceId: string | null;
  fixedWildcardValues: string[] | null;
  isFetchingFixedWildcardValues: boolean;
  onSelectRange: (range: PromptRange) => void;
  onRemoveWildcard: (occurrence: PromptWildcardOccurrence) => void;
  onRandomWildcard: (occurrence: PromptWildcardOccurrence) => void;
  onPickFixedWildcard: (occurrence: PromptWildcardOccurrence) => void;
  onCyclicWildcard: (occurrence: PromptWildcardOccurrence) => void;
  onExploreAllWildcard: (occurrence: PromptWildcardOccurrence) => void;
  onFixedValue: (value: string) => void;
  onAdjustWeight: (occurrence: PromptWeightOccurrence, direction: 'increment' | 'decrement') => void;
};

const ACTION_BUTTON_SIZE = 7;

export const PromptInspector = memo(
  ({
    occurrences,
    randomRefreshMode,
    fixedWildcardOccurrenceId,
    fixedWildcardValues,
    isFetchingFixedWildcardValues,
    onSelectRange,
    onRemoveWildcard,
    onRandomWildcard,
    onPickFixedWildcard,
    onCyclicWildcard,
    onExploreAllWildcard,
    onFixedValue,
    onAdjustWeight,
  }: PromptInspectorProps) => {
    if (occurrences.length === 0) {
      return null;
    }

    return (
      <Flex
        flexDir="column"
        borderWidth={1}
        borderColor="base.700"
        borderRadius="base"
        overflow="hidden"
        data-testid="prompt-workbench-inspector"
      >
        {occurrences.map((occurrence) =>
          occurrence.type === 'wildcard' ? (
            <WildcardInspectorRow
              key={occurrence.id}
              occurrence={occurrence}
              randomRefreshMode={randomRefreshMode}
              fixedValues={fixedWildcardOccurrenceId === occurrence.id ? fixedWildcardValues : null}
              isFetchingFixedValues={fixedWildcardOccurrenceId === occurrence.id && isFetchingFixedWildcardValues}
              onSelectRange={onSelectRange}
              onRemoveWildcard={onRemoveWildcard}
              onRandomWildcard={onRandomWildcard}
              onPickFixedWildcard={onPickFixedWildcard}
              onCyclicWildcard={onCyclicWildcard}
              onExploreAllWildcard={onExploreAllWildcard}
              onFixedValue={onFixedValue}
            />
          ) : (
            <WeightInspectorRow
              key={occurrence.id}
              occurrence={occurrence}
              onSelectRange={onSelectRange}
              onAdjustWeight={onAdjustWeight}
            />
          )
        )}
      </Flex>
    );
  }
);

PromptInspector.displayName = 'PromptInspector';

type WildcardInspectorRowProps = {
  occurrence: PromptWildcardOccurrence;
  randomRefreshMode: 'manual' | 'per_enqueue';
  fixedValues: string[] | null;
  isFetchingFixedValues: boolean;
  onSelectRange: (range: PromptRange) => void;
  onRemoveWildcard: (occurrence: PromptWildcardOccurrence) => void;
  onRandomWildcard: (occurrence: PromptWildcardOccurrence) => void;
  onPickFixedWildcard: (occurrence: PromptWildcardOccurrence) => void;
  onCyclicWildcard: (occurrence: PromptWildcardOccurrence) => void;
  onExploreAllWildcard: (occurrence: PromptWildcardOccurrence) => void;
  onFixedValue: (value: string) => void;
};

const WildcardInspectorRow = memo(
  ({
    occurrence,
    randomRefreshMode,
    fixedValues,
    isFetchingFixedValues,
    onSelectRange,
    onRemoveWildcard,
    onRandomWildcard,
    onPickFixedWildcard,
    onCyclicWildcard,
    onExploreAllWildcard,
    onFixedValue,
  }: WildcardInspectorRowProps) => {
    const hasWildcard = occurrence.wildcard !== null;
    const isActionable = occurrence.behavior !== 'missing' && occurrence.behavior !== 'unavailable';

    const onSelectMouseDown = useCallback(
      (e: MouseEvent<HTMLButtonElement>) => {
        e.preventDefault();
        onSelectRange(occurrence.range);
      },
      [occurrence.range, onSelectRange]
    );

    const onRemoveMouseDown = useCallback(
      (e: MouseEvent<HTMLButtonElement>) => {
        e.preventDefault();
        onRemoveWildcard(occurrence);
      },
      [occurrence, onRemoveWildcard]
    );

    const onRandomMouseDown = useCallback(
      (e: MouseEvent<HTMLButtonElement>) => {
        e.preventDefault();
        onRandomWildcard(occurrence);
      },
      [occurrence, onRandomWildcard]
    );

    const onPickFixedMouseDown = useCallback(
      (e: MouseEvent<HTMLButtonElement>) => {
        e.preventDefault();
        onPickFixedWildcard(occurrence);
      },
      [occurrence, onPickFixedWildcard]
    );

    const onCycleMouseDown = useCallback(
      (e: MouseEvent<HTMLButtonElement>) => {
        e.preventDefault();
        onCyclicWildcard(occurrence);
      },
      [occurrence, onCyclicWildcard]
    );

    const onAllMouseDown = useCallback(
      (e: MouseEvent<HTMLButtonElement>) => {
        e.preventDefault();
        onExploreAllWildcard(occurrence);
      },
      [occurrence, onExploreAllWildcard]
    );

    return (
      <Box>
        <Flex alignItems="center" gap={1} px={1} minH={8} _hover={{ bg: 'base.800' }}>
          <Tooltip label={occurrence.token}>
            <Button
              size="xs"
              variant="ghost"
              justifyContent="flex-start"
              flexGrow={1}
              minW={0}
              h={7}
              px={2}
              onMouseDown={onSelectMouseDown}
            >
              <Text as="span" noOfLines={1} color="base.100" fontWeight="semibold">
                {occurrence.path}
              </Text>
            </Button>
          </Tooltip>
          <Badge size="sm" colorScheme={getWildcardBadgeColorScheme(occurrence)}>
            {getWildcardBehaviorLabel(occurrence, randomRefreshMode)}
          </Badge>
          <Text fontSize="xs" color="base.400" textAlign="end" w={12} flexShrink={0} noOfLines={1}>
            {occurrence.valueCount === null ? '' : `${occurrence.valueCount} values`}
          </Text>
          <Flex gap={0.5} flexShrink={0}>
            {isActionable && (
              <>
                <IconButton
                  aria-label={`Set ${occurrence.path} to random every Invoke`}
                  tooltip="Random every Invoke"
                  size="xs"
                  variant="ghost"
                  minW={ACTION_BUTTON_SIZE}
                  h={ACTION_BUTTON_SIZE}
                  icon={<PiDiceFiveBold />}
                  onMouseDown={onRandomMouseDown}
                />
                {hasWildcard && (
                  <IconButton
                    aria-label={`Pick a fixed value for ${occurrence.path}`}
                    tooltip="Pick fixed value"
                    size="xs"
                    variant="ghost"
                    minW={ACTION_BUTTON_SIZE}
                    h={ACTION_BUTTON_SIZE}
                    icon={<PiPushPinSimpleBold />}
                    onMouseDown={onPickFixedMouseDown}
                  />
                )}
                <IconButton
                  aria-label={`Set ${occurrence.path} to cycle`}
                  tooltip="Cycle"
                  size="xs"
                  variant="ghost"
                  minW={ACTION_BUTTON_SIZE}
                  h={ACTION_BUTTON_SIZE}
                  icon={<PiRepeatBold />}
                  onMouseDown={onCycleMouseDown}
                />
                <IconButton
                  aria-label={`Explore all combinations for ${occurrence.path}`}
                  tooltip="Explore all"
                  size="xs"
                  variant="ghost"
                  minW={ACTION_BUTTON_SIZE}
                  h={ACTION_BUTTON_SIZE}
                  icon={<PiSquaresFourBold />}
                  onMouseDown={onAllMouseDown}
                />
              </>
            )}
            <IconButton
              aria-label={`Select ${occurrence.path} in prompt`}
              tooltip="Select in prompt"
              size="xs"
              variant="ghost"
              minW={ACTION_BUTTON_SIZE}
              h={ACTION_BUTTON_SIZE}
              icon={<PiSelectionBold />}
              onMouseDown={onSelectMouseDown}
            />
            <IconButton
              aria-label={`Remove ${occurrence.path} from prompt`}
              tooltip="Remove"
              size="xs"
              variant="ghost"
              colorScheme="error"
              minW={ACTION_BUTTON_SIZE}
              h={ACTION_BUTTON_SIZE}
              icon={<PiXBold />}
              onMouseDown={onRemoveMouseDown}
            />
          </Flex>
        </Flex>
        {(fixedValues || isFetchingFixedValues) && (
          <Flex flexDir="column" ps={4} pe={1} pb={1} gap={0.5}>
            {isFetchingFixedValues && (
              <Text fontSize="xs" color="base.400">
                Loading values...
              </Text>
            )}
            {fixedValues?.map((value) => (
              <FixedValueButton key={value} value={value} onFixedValue={onFixedValue} />
            ))}
          </Flex>
        )}
      </Box>
    );
  }
);

WildcardInspectorRow.displayName = 'WildcardInspectorRow';

const FixedValueButton = memo(
  ({ value, onFixedValue }: { value: string; onFixedValue: (value: string) => void }) => {
    const onMouseDown = useCallback(
      (e: MouseEvent<HTMLButtonElement>) => {
        e.preventDefault();
        onFixedValue(value);
      },
      [onFixedValue, value]
    );

    return (
      <Button size="xs" variant="ghost" justifyContent="flex-start" h={7} px={2} onMouseDown={onMouseDown}>
        <Text as="span" noOfLines={1}>
          {value}
        </Text>
      </Button>
    );
  }
);

FixedValueButton.displayName = 'FixedValueButton';

type WeightInspectorRowProps = {
  occurrence: PromptWeightOccurrence;
  onSelectRange: (range: PromptRange) => void;
  onAdjustWeight: (occurrence: PromptWeightOccurrence, direction: 'increment' | 'decrement') => void;
};

const WeightInspectorRow = memo(({ occurrence, onSelectRange, onAdjustWeight }: WeightInspectorRowProps) => {
  const onSelectMouseDown = useCallback(
    (e: MouseEvent<HTMLButtonElement>) => {
      e.preventDefault();
      onSelectRange(occurrence.range);
    },
    [occurrence.range, onSelectRange]
  );

  const onDecrementMouseDown = useCallback(
    (e: MouseEvent<HTMLButtonElement>) => {
      e.preventDefault();
      onAdjustWeight(occurrence, 'decrement');
    },
    [occurrence, onAdjustWeight]
  );

  const onIncrementMouseDown = useCallback(
    (e: MouseEvent<HTMLButtonElement>) => {
      e.preventDefault();
      onAdjustWeight(occurrence, 'increment');
    },
    [occurrence, onAdjustWeight]
  );

  return (
    <Flex alignItems="center" gap={1} px={1} minH={8} _hover={{ bg: 'base.800' }}>
      <Tooltip label={String(occurrence.attention)}>
        <Button
          size="xs"
          variant="ghost"
          justifyContent="flex-start"
          flexGrow={1}
          minW={0}
          h={7}
          px={2}
          onMouseDown={onSelectMouseDown}
        >
          <Text as="span" noOfLines={1} color="base.100" fontWeight="semibold">
            {occurrence.text}
          </Text>
        </Button>
      </Tooltip>
      <Badge size="sm" colorScheme={occurrence.isSupported ? 'green' : 'yellow'}>
        {getWeightBehaviorLabel(occurrence)}
      </Badge>
      <Flex gap={0.5} flexShrink={0}>
        <Tooltip label="Decrease weight">
          <Button size="xs" variant="outline" minW={ACTION_BUTTON_SIZE} h={ACTION_BUTTON_SIZE} onMouseDown={onDecrementMouseDown}>
            -
          </Button>
        </Tooltip>
        <Tooltip label="Increase weight">
          <Button size="xs" variant="outline" minW={ACTION_BUTTON_SIZE} h={ACTION_BUTTON_SIZE} onMouseDown={onIncrementMouseDown}>
            +
          </Button>
        </Tooltip>
        <IconButton
          aria-label={`Select ${occurrence.text} in prompt`}
          tooltip="Select in prompt"
          size="xs"
          variant="ghost"
          minW={ACTION_BUTTON_SIZE}
          h={ACTION_BUTTON_SIZE}
          icon={<PiSelectionBold />}
          onMouseDown={onSelectMouseDown}
        />
      </Flex>
    </Flex>
  );
});

WeightInspectorRow.displayName = 'WeightInspectorRow';

const getWildcardBadgeColorScheme = (occurrence: PromptWildcardOccurrence): string => {
  switch (occurrence.behavior) {
    case 'random':
    case 'cycle':
    case 'all':
      return 'green';
    case 'missing':
    case 'unavailable':
      return 'red';
  }
};
