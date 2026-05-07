import { Box, Button, Flex, Text, Tooltip } from '@invoke-ai/ui-library';
import type { DynamicPromptRandomRefreshMode } from 'features/dynamicPrompts/store/dynamicPromptsSlice';
import type { MouseEvent } from 'react';
import { memo, useCallback } from 'react';

import {
  getWeightBehaviorLabel,
  getWeightShortLabel,
  getWildcardBehaviorIconType,
  getWildcardBehaviorLabel,
  getWildcardBehaviorShortLabel,
  type PromptRange,
  type PromptWeightOccurrence,
  type PromptWildcardOccurrence,
  type PromptWorkbenchOccurrence,
  type WildcardBehaviorAction,
} from './occurrences';
import { PromptWildcardBehaviorMenu } from './PromptWildcardBehaviorMenu';
import { PromptWorkbenchBadge, type PromptWorkbenchBadgeTone } from './PromptWorkbenchBadge';

type PromptInspectorProps = {
  occurrences: PromptWorkbenchOccurrence[];
  randomRefreshMode: DynamicPromptRandomRefreshMode;
  fixedWildcardOccurrenceId: string | null;
  fixedWildcardValues: string[] | null;
  isFetchingFixedWildcardValues: boolean;
  activeFixedValueIndex: number;
  onSelectRange: (range: PromptRange) => void;
  onWildcardBehaviorAction: (occurrence: PromptWildcardOccurrence, action: WildcardBehaviorAction) => void;
  onFixedValue: (value: string) => void;
  setFixedValueElement: (index: number, element: HTMLElement | null) => void;
};

export const PromptInspector = memo(
  ({
    occurrences,
    randomRefreshMode,
    fixedWildcardOccurrenceId,
    fixedWildcardValues,
    isFetchingFixedWildcardValues,
    activeFixedValueIndex,
    onSelectRange,
    onWildcardBehaviorAction,
    onFixedValue,
    setFixedValueElement,
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
              activeFixedValueIndex={activeFixedValueIndex}
              onSelectRange={onSelectRange}
              onWildcardBehaviorAction={onWildcardBehaviorAction}
              onFixedValue={onFixedValue}
              setFixedValueElement={setFixedValueElement}
            />
          ) : (
            <WeightInspectorRow
              key={occurrence.id}
              occurrence={occurrence}
              onSelectRange={onSelectRange}
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
  randomRefreshMode: DynamicPromptRandomRefreshMode;
  fixedValues: string[] | null;
  isFetchingFixedValues: boolean;
  activeFixedValueIndex: number;
  onSelectRange: (range: PromptRange) => void;
  onWildcardBehaviorAction: (occurrence: PromptWildcardOccurrence, action: WildcardBehaviorAction) => void;
  onFixedValue: (value: string) => void;
  setFixedValueElement: (index: number, element: HTMLElement | null) => void;
};

const WildcardInspectorRow = memo(
  ({
    occurrence,
    randomRefreshMode,
    fixedValues,
    isFetchingFixedValues,
    activeFixedValueIndex,
    onSelectRange,
    onWildcardBehaviorAction,
    onFixedValue,
    setFixedValueElement,
  }: WildcardInspectorRowProps) => {
    const hasWildcard = occurrence.wildcard !== null;
    const isActionable = occurrence.behavior !== 'missing' && occurrence.behavior !== 'unavailable';
    const selectionRange = occurrence.weight?.range ?? occurrence.range;

    const onSelectMouseDown = useCallback(
      (e: MouseEvent<HTMLButtonElement>) => {
        e.preventDefault();
        onSelectRange(selectionRange);
      },
      [onSelectRange, selectionRange]
    );

    const onBehaviorAction = useCallback(
      (action: WildcardBehaviorAction) => {
        onWildcardBehaviorAction(occurrence, action);
      },
      [occurrence, onWildcardBehaviorAction]
    );

    return (
      <Box>
        <Flex alignItems="center" gap={1} px={1} minH={8} role="group" _hover={{ bg: 'base.800' }}>
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
              <Text as="span" noOfLines={1} color="base.100" fontSize="sm" fontWeight="semibold">
                {occurrence.path}
              </Text>
            </Button>
          </Tooltip>
          <Flex gap={0.5} flexShrink={0} minW={0}>
            <Tooltip label={getWildcardBehaviorLabel(occurrence, randomRefreshMode)}>
              <PromptWorkbenchBadge tone={getWildcardBadgeTone(occurrence)}>
                {getWildcardBehaviorShortLabel(occurrence, randomRefreshMode)}
              </PromptWorkbenchBadge>
            </Tooltip>
            {occurrence.valueCount !== null && (
              <Tooltip label={`${occurrence.valueCount} values`}>
                <PromptWorkbenchBadge>
                  {occurrence.valueCount}
                </PromptWorkbenchBadge>
              </Tooltip>
            )}
            {occurrence.weight && (
              <Tooltip label={getWeightBehaviorLabel(occurrence.weight)}>
                <PromptWorkbenchBadge tone={occurrence.weight.isSupported ? 'neutral' : 'warning'}>
                  {getWeightShortLabel(occurrence.weight)}
                </PromptWorkbenchBadge>
              </Tooltip>
            )}
          </Flex>
          <Flex gap={0.5} flexShrink={0} alignItems="center">
            <PromptWildcardBehaviorMenu
              ariaLabel={`Change ${occurrence.path} wildcard behavior`}
              tooltip="Wildcard behavior"
              iconType={getWildcardBehaviorIconType(occurrence)}
              isActionable={isActionable}
              canPickFixedValue={hasWildcard}
              includeRemove
              onAction={onBehaviorAction}
            />
          </Flex>
        </Flex>
        {(fixedValues || isFetchingFixedValues) && (
          <Flex flexDir="column" ps={4} pe={1} pb={1} gap={0.5} maxH={36} overflowY="auto">
            {isFetchingFixedValues && (
              <Text fontSize="xs" color="base.400">
                Loading values...
              </Text>
            )}
            {fixedValues?.map((value, index) => (
              <FixedValueButton
                key={value}
                value={value}
                index={index}
                isActive={activeFixedValueIndex === index}
                onFixedValue={onFixedValue}
                setFixedValueElement={setFixedValueElement}
              />
            ))}
          </Flex>
        )}
      </Box>
    );
  }
);

WildcardInspectorRow.displayName = 'WildcardInspectorRow';

const FixedValueButton = memo(
  ({
    value,
    index,
    isActive,
    onFixedValue,
    setFixedValueElement,
  }: {
    value: string;
    index: number;
    isActive: boolean;
    onFixedValue: (value: string) => void;
    setFixedValueElement: (index: number, element: HTMLElement | null) => void;
  }) => {
    const onMouseDown = useCallback(
      (e: MouseEvent<HTMLButtonElement>) => {
        e.preventDefault();
        onFixedValue(value);
      },
      [onFixedValue, value]
    );

    const setElement = useCallback(
      (element: HTMLElement | null) => {
        setFixedValueElement(index, element);
      },
      [index, setFixedValueElement]
    );

    return (
      <Button
        ref={setElement}
        size="xs"
        variant="ghost"
        justifyContent="flex-start"
        h={7}
        px={2}
        bg={isActive ? 'base.700' : undefined}
        onMouseDown={onMouseDown}
      >
        <Text as="span" noOfLines={1} fontSize="sm">
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
};

const WeightInspectorRow = memo(
  ({ occurrence, onSelectRange }: WeightInspectorRowProps) => {
    const onSelectMouseDown = useCallback(
      (e: MouseEvent<HTMLButtonElement>) => {
        e.preventDefault();
        onSelectRange(occurrence.range);
      },
      [occurrence.range, onSelectRange]
    );

    return (
      <Flex alignItems="center" gap={1} px={1} minH={8} role="group" _hover={{ bg: 'base.800' }}>
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
            <Text as="span" noOfLines={1} color="base.100" fontSize="sm" fontWeight="semibold">
              {occurrence.text}
            </Text>
          </Button>
        </Tooltip>
        <Tooltip label={getWeightBehaviorLabel(occurrence)}>
          <PromptWorkbenchBadge tone={occurrence.isSupported ? 'neutral' : 'warning'}>
            {getWeightShortLabel(occurrence)}
          </PromptWorkbenchBadge>
        </Tooltip>
      </Flex>
    );
  }
);

WeightInspectorRow.displayName = 'WeightInspectorRow';

const getWildcardBadgeTone = (occurrence: PromptWildcardOccurrence): PromptWorkbenchBadgeTone => {
  switch (occurrence.behavior) {
    case 'random':
    case 'cycle':
    case 'all':
      return 'neutral';
    case 'missing':
    case 'unavailable':
      return 'error';
  }
};
