import {
  Box,
  Button,
  Flex,
  IconButton,
  Menu,
  MenuButton,
  MenuItem,
  MenuList,
  Text,
  Tooltip,
} from '@invoke-ai/ui-library';
import type { DynamicPromptRandomRefreshMode } from 'features/dynamicPrompts/store/dynamicPromptsSlice';
import type { MouseEvent, ReactElement, ReactNode } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import {
  PiCaretDownBold,
  PiDiceFiveBold,
  PiDotsThreeVerticalBold,
  PiPushPinSimpleBold,
  PiRepeatBold,
  PiScalesBold,
  PiTrashSimpleBold,
} from 'react-icons/pi';

import { type PromptWorkbenchTranslation, tx } from './i18n';
import {
  getWeightBehaviorLabel,
  getWeightShortLabel,
  getWildcardBehaviorLabel,
  getWildcardBehaviorShortLabel,
  type PromptRange,
  type PromptWeightOccurrence,
  type PromptWildcardOccurrence,
  type PromptWorkbenchOccurrence,
  type WildcardBehaviorAction,
} from './occurrences';
import type { PromptWorkbenchBadgeTone } from './PromptWorkbenchBadge';

type PromptInspectorProps = {
  occurrences: PromptWorkbenchOccurrence[];
  randomRefreshMode: DynamicPromptRandomRefreshMode;
  fixedWildcardOccurrenceId: string | null;
  fixedWildcardValues: string[] | null;
  isFetchingFixedWildcardValues: boolean;
  activeFixedValueIndex: number;
  onSelectRange: (range: PromptRange) => void;
  onWildcardBehaviorAction: (occurrence: PromptWildcardOccurrence, action: WildcardBehaviorAction) => void;
  onRemoveWeightOccurrence: (occurrence: PromptWeightOccurrence) => void;
  onFixedValue: (value: string) => void;
  setFixedValueElement: (index: number, element: HTMLElement | null) => void;
};

const ROW_GRID_TEMPLATE = '2.75rem minmax(0, 1fr) minmax(6.75rem, 7.5rem) 2.5rem 2.25rem';
const INTENT_CARD_BG = 'linear-gradient(180deg, rgba(25, 35, 45, 0.94) 0%, rgba(17, 26, 35, 0.96) 100%)';
const INTENT_CARD_HOVER_BG = 'linear-gradient(180deg, rgba(29, 41, 53, 0.98) 0%, rgba(20, 31, 41, 0.98) 100%)';
const INTENT_CARD_BORDER = 'rgba(126, 143, 164, 0.3)';
const INTENT_CARD_BORDER_HOVER = 'rgba(153, 170, 191, 0.42)';
const INTENT_CARD_ERROR_BORDER = 'rgba(248, 113, 113, 0.5)';
const INTENT_CARD_ERROR_BORDER_HOVER = 'rgba(252, 165, 165, 0.58)';

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
    onRemoveWeightOccurrence,
    onFixedValue,
    setFixedValueElement,
  }: PromptInspectorProps) => {
    if (occurrences.length === 0) {
      return null;
    }

    return (
      <Flex flexDir="column" gap={1.5} data-testid="prompt-workbench-inspector">
        <Flex flexDir="column" gap={1.5}>
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
                onRemoveWeightOccurrence={onRemoveWeightOccurrence}
              />
            )
          )}
        </Flex>
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
    const { t } = useTranslation();
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
      <IntentRowBox markerColor={getWildcardMarkerColor(occurrence)} tone={getWildcardRowTone(occurrence)}>
        <Box
          display="grid"
          gridTemplateColumns={ROW_GRID_TEMPLATE}
          alignItems="stretch"
          minH={16}
          role="group"
          aria-label={t('promptWorkbench.rows.wildcardIntentAria', { path: occurrence.path })}
        >
          <TypeCell color={getWildcardIconColor(occurrence)} icon={<PiDiceFiveBold />} />
          <LabelCell>
            <Tooltip label={occurrence.token}>
              <Button
                size="sm"
                variant="ghost"
                justifyContent="flex-start"
                minW={0}
                h="auto"
                minH={6}
                px={0}
                py={0}
                _hover={{ bg: 'transparent' }}
                onMouseDown={onSelectMouseDown}
              >
                <Text as="span" noOfLines={1} color="base.100" fontSize="sm" fontWeight="semibold">
                  {occurrence.path}
                </Text>
              </Button>
            </Tooltip>
            <Text color="base.400" fontSize="xs" fontWeight="medium" noOfLines={1}>
              {t(getWildcardSecondaryText(occurrence).key, getWildcardSecondaryText(occurrence).options)}
            </Text>
          </LabelCell>
          <BehaviorCell>
            <WildcardBehaviorControl
              occurrence={occurrence}
              randomRefreshMode={randomRefreshMode}
              isActionable={isActionable}
              canPickFixedValue={hasWildcard}
              onAction={onBehaviorAction}
            />
          </BehaviorCell>
          <WeightCell>
            {occurrence.weight ? (
              <WeightValue occurrence={occurrence.weight} />
            ) : (
              <Text color="base.400" fontSize="md" fontWeight="semibold">
                -
              </Text>
            )}
          </WeightCell>
          <ActionsCell>
            <WildcardActionsMenu occurrence={occurrence} onAction={onBehaviorAction} />
          </ActionsCell>
        </Box>
        {(fixedValues || isFetchingFixedValues) && (
          <Flex flexDir="column" ms={12} me={2} mb={2} gap={0.5} maxH={36} overflowY="auto">
            {isFetchingFixedValues && (
              <Text fontSize="sm" color="base.400">
                {t('promptWorkbench.values.loading')}
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
      </IntentRowBox>
    );
  }
);

WildcardInspectorRow.displayName = 'WildcardInspectorRow';

const WildcardBehaviorControl = memo(
  ({
    occurrence,
    randomRefreshMode,
    isActionable,
    canPickFixedValue,
    onAction,
  }: {
    occurrence: PromptWildcardOccurrence;
    randomRefreshMode: DynamicPromptRandomRefreshMode;
    isActionable: boolean;
    canPickFixedValue: boolean;
    onAction: (action: WildcardBehaviorAction) => void;
  }) => {
    const { t } = useTranslation();
    const onButtonMouseDown = useCallback((e: MouseEvent<HTMLButtonElement>) => {
      e.preventDefault();
    }, []);

    const onRandom = useCallback(() => {
      onAction('random');
    }, [onAction]);

    const onCycle = useCallback(() => {
      onAction('cycle');
    }, [onAction]);

    const onFixed = useCallback(() => {
      onAction('fixed');
    }, [onAction]);

    if (!isActionable) {
      return (
        <Tooltip
          label={t(
            getWildcardBehaviorLabel(occurrence, randomRefreshMode).key,
            getWildcardBehaviorLabel(occurrence, randomRefreshMode).options
          )}
        >
          <Text color="error.300" fontSize="md" fontWeight="semibold" noOfLines={1}>
            {t(
              getWildcardBehaviorShortLabel(occurrence, randomRefreshMode).key,
              getWildcardBehaviorShortLabel(occurrence, randomRefreshMode).options
            )}
          </Text>
        </Tooltip>
      );
    }

    return (
      <Menu>
        <Tooltip
          label={t(
            getWildcardBehaviorLabel(occurrence, randomRefreshMode).key,
            getWildcardBehaviorLabel(occurrence, randomRefreshMode).options
          )}
        >
          <MenuButton
            as={Button}
            size="sm"
            variant="outline"
            h={8}
            minW={0}
            w="full"
            px={2}
            rightIcon={<PiCaretDownBold />}
            onMouseDown={onButtonMouseDown}
          >
            <Text as="span" noOfLines={1} fontSize="xs" fontWeight="medium">
              {t(
                getWildcardBehaviorShortLabel(occurrence, randomRefreshMode).key,
                getWildcardBehaviorShortLabel(occurrence, randomRefreshMode).options
              )}
            </Text>
          </MenuButton>
        </Tooltip>
        <MenuList>
          <MenuItem
            icon={<PiDiceFiveBold />}
            onClick={onRandom}
            title={t('promptWorkbench.actions.randomWildcardTooltip')}
          >
            {t('promptWorkbench.actions.randomWildcard')}
          </MenuItem>
          <MenuItem icon={<PiRepeatBold />} onClick={onCycle} title={t('promptWorkbench.actions.cycleTooltip')}>
            {t('promptWorkbench.actions.cycle')}
          </MenuItem>
          <MenuItem icon={<PiPushPinSimpleBold />} onClick={onFixed} isDisabled={!canPickFixedValue}>
            {t('promptWorkbench.actions.pickFixed')}
          </MenuItem>
        </MenuList>
      </Menu>
    );
  }
);

WildcardBehaviorControl.displayName = 'WildcardBehaviorControl';

const WildcardActionsMenu = memo(
  ({
    occurrence,
    onAction,
  }: {
    occurrence: PromptWildcardOccurrence;
    onAction: (action: WildcardBehaviorAction) => void;
  }) => {
    const { t } = useTranslation();
    const onButtonMouseDown = useCallback((e: MouseEvent<HTMLButtonElement>) => {
      e.preventDefault();
    }, []);

    const onRemove = useCallback(() => {
      onAction('remove');
    }, [onAction]);

    return (
      <Menu>
        <MenuButton
          as={IconButton}
          aria-label={t('promptWorkbench.actions.openWildcardActionsAria', { path: occurrence.path })}
          tooltip={t('promptWorkbench.actions.wildcardActions')}
          size="sm"
          variant="ghost"
          minW={7}
          h={7}
          color="base.300"
          icon={<PiDotsThreeVerticalBold />}
          onMouseDown={onButtonMouseDown}
        />
        <MenuList>
          <MenuItem icon={<PiTrashSimpleBold />} color="error.300" onClick={onRemove}>
            {t('promptWorkbench.actions.remove')}
          </MenuItem>
        </MenuList>
      </Menu>
    );
  }
);

WildcardActionsMenu.displayName = 'WildcardActionsMenu';

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
        size="sm"
        variant="ghost"
        justifyContent="flex-start"
        h={8}
        px={3}
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
  onRemoveWeightOccurrence: (occurrence: PromptWeightOccurrence) => void;
};

const WeightInspectorRow = memo(({ occurrence, onSelectRange, onRemoveWeightOccurrence }: WeightInspectorRowProps) => {
  const { t } = useTranslation();
  const onSelectMouseDown = useCallback(
    (e: MouseEvent<HTMLButtonElement>) => {
      e.preventDefault();
      onSelectRange(occurrence.range);
    },
    [occurrence.range, onSelectRange]
  );

  const onRemove = useCallback(() => {
    onRemoveWeightOccurrence(occurrence);
  }, [occurrence, onRemoveWeightOccurrence]);

  return (
    <IntentRowBox markerColor={occurrence.isSupported ? 'warning.500' : 'warning.600'}>
      <Box
        display="grid"
        gridTemplateColumns={ROW_GRID_TEMPLATE}
        alignItems="stretch"
        minH={16}
        role="group"
        aria-label={t('promptWorkbench.rows.weightIntentAria', { text: occurrence.text })}
      >
        <TypeCell color={occurrence.isSupported ? 'warning.300' : 'warning.400'} icon={<PiScalesBold />} />
        <LabelCell>
          <Tooltip label={String(occurrence.attention)}>
            <Button
              size="sm"
              variant="ghost"
              justifyContent="flex-start"
              minW={0}
              h="auto"
              minH={6}
              px={0}
              py={0}
              _hover={{ bg: 'transparent' }}
              onMouseDown={onSelectMouseDown}
            >
              <Text as="span" noOfLines={1} color="base.100" fontSize="sm" fontWeight="semibold">
                {occurrence.text}
              </Text>
            </Button>
          </Tooltip>
          <Text color="base.400" fontSize="xs" fontWeight="medium" noOfLines={1}>
            {t('promptWorkbench.weight.promptWeight')}
          </Text>
        </LabelCell>
        <BehaviorCell>
          <Box />
        </BehaviorCell>
        <WeightCell>
          <WeightValue occurrence={occurrence} />
        </WeightCell>
        <ActionsCell>
          <WeightActionsMenu occurrence={occurrence} onRemove={onRemove} />
        </ActionsCell>
      </Box>
    </IntentRowBox>
  );
});

WeightInspectorRow.displayName = 'WeightInspectorRow';

const WeightActionsMenu = memo(
  ({ occurrence, onRemove }: { occurrence: PromptWeightOccurrence; onRemove: () => void }) => {
    const { t } = useTranslation();
    const onButtonMouseDown = useCallback((e: MouseEvent<HTMLButtonElement>) => {
      e.preventDefault();
    }, []);

    return (
      <Menu>
        <MenuButton
          as={IconButton}
          aria-label={t('promptWorkbench.actions.openWeightActionsAria', { text: occurrence.text })}
          tooltip={t('promptWorkbench.actions.weightActions')}
          size="sm"
          variant="ghost"
          minW={7}
          h={7}
          color="base.300"
          icon={<PiDotsThreeVerticalBold />}
          onMouseDown={onButtonMouseDown}
        />
        <MenuList>
          <MenuItem icon={<PiTrashSimpleBold />} color="error.300" onClick={onRemove}>
            {t('promptWorkbench.actions.remove')}
          </MenuItem>
        </MenuList>
      </Menu>
    );
  }
);

WeightActionsMenu.displayName = 'WeightActionsMenu';

const WeightValue = memo(({ occurrence }: { occurrence: PromptWeightOccurrence }) => {
  const { t } = useTranslation();
  const behaviorLabel = getWeightBehaviorLabel(occurrence);
  const shortLabel = getWeightShortLabel(occurrence);

  return (
    <Tooltip label={t(behaviorLabel.key, behaviorLabel.options)}>
      <Text
        color={occurrence.isSupported ? 'base.100' : 'warning.300'}
        fontSize="md"
        fontWeight="semibold"
        lineHeight="short"
        noOfLines={1}
      >
        {t(shortLabel.key, shortLabel.options)}
      </Text>
    </Tooltip>
  );
});

WeightValue.displayName = 'WeightValue';

const TypeCell = memo(({ color, icon }: { color: string; icon: ReactElement }) => (
  <Flex alignItems="center" justifyContent="center" color={color} fontSize="1.2rem" minW={0}>
    {icon}
  </Flex>
));

TypeCell.displayName = 'TypeCell';

const LabelCell = memo(({ children }: { children: ReactNode }) => (
  <Flex flexDir="column" justifyContent="center" gap={0.5} minW={0} pe={2}>
    {children}
  </Flex>
));

LabelCell.displayName = 'LabelCell';

const BehaviorCell = memo(({ children }: { children: ReactNode }) => (
  <Flex alignItems="center" justifyContent="center" minW={0} px={1}>
    {children}
  </Flex>
));

BehaviorCell.displayName = 'BehaviorCell';

const WeightCell = memo(({ children }: { children: ReactNode }) => (
  <Flex alignItems="center" justifyContent="center" minW={0} px={1} borderLeftWidth={1} borderLeftColor="base.700">
    {children}
  </Flex>
));

WeightCell.displayName = 'WeightCell';

const ActionsCell = memo(({ children }: { children: ReactNode }) => (
  <Flex alignItems="center" justifyContent="center" minW={0} px={1} borderLeftWidth={1} borderLeftColor="base.700">
    {children}
  </Flex>
));

ActionsCell.displayName = 'ActionsCell';

const IntentRowBox = memo(
  ({
    children,
    markerColor,
    tone = 'neutral',
  }: {
    children: ReactNode;
    markerColor: string;
    tone?: PromptWorkbenchBadgeTone;
  }) => (
    <Box
      position="relative"
      borderWidth={1}
      borderColor={tone === 'error' ? INTENT_CARD_ERROR_BORDER : INTENT_CARD_BORDER}
      borderRadius="base"
      bg={INTENT_CARD_BG}
      boxShadow="0 1px 0 rgba(255, 255, 255, 0.025) inset, 0 16px 34px rgba(0, 0, 0, 0.12)"
      overflow="hidden"
      _before={{
        content: '""',
        position: 'absolute',
        left: 0,
        top: 0,
        bottom: 0,
        w: 1.5,
        bg: markerColor,
      }}
      _hover={{
        bg: INTENT_CARD_HOVER_BG,
        borderColor: tone === 'error' ? INTENT_CARD_ERROR_BORDER_HOVER : INTENT_CARD_BORDER_HOVER,
      }}
    >
      {children}
    </Box>
  )
);

IntentRowBox.displayName = 'IntentRowBox';

const getWildcardSecondaryText = (occurrence: PromptWildcardOccurrence): PromptWorkbenchTranslation => {
  if (occurrence.behavior === 'missing') {
    return tx('promptWorkbench.rows.missingWildcard');
  }
  if (occurrence.behavior === 'unavailable') {
    return tx('promptWorkbench.rows.wildcardIndexUnavailable');
  }

  const options = occurrence.valueCount !== null ? { count: occurrence.valueCount } : undefined;

  switch (occurrence.behavior) {
    case 'random':
      return tx(
        occurrence.valueCount !== null
          ? 'promptWorkbench.rows.randomWildcardWithCount'
          : 'promptWorkbench.rows.randomWildcard',
        options
      );
    case 'cycle':
      return tx(
        occurrence.valueCount !== null ? 'promptWorkbench.rows.cycleWithCount' : 'promptWorkbench.rows.cycle',
        options
      );
    case 'all':
      return tx(
        occurrence.valueCount !== null
          ? 'promptWorkbench.rows.allCombinationsWithCount'
          : 'promptWorkbench.rows.allCombinations',
        options
      );
  }
};

const getWildcardIconColor = (occurrence: PromptWildcardOccurrence): string => {
  switch (occurrence.behavior) {
    case 'random':
    case 'cycle':
    case 'all':
      return occurrence.weight?.isSupported === false ? 'warning.300' : 'base.300';
    case 'missing':
    case 'unavailable':
      return 'error.300';
  }
};

const getWildcardMarkerColor = (occurrence: PromptWildcardOccurrence): string => {
  switch (occurrence.behavior) {
    case 'random':
    case 'cycle':
    case 'all':
      return occurrence.weight?.isSupported === false ? 'warning.500' : 'invokeBlue.400';
    case 'missing':
    case 'unavailable':
      return 'error.500';
  }
};

const getWildcardRowTone = (occurrence: PromptWildcardOccurrence): PromptWorkbenchBadgeTone => {
  switch (occurrence.behavior) {
    case 'missing':
    case 'unavailable':
      return 'error';
    case 'random':
    case 'cycle':
    case 'all':
      return 'neutral';
  }
};
