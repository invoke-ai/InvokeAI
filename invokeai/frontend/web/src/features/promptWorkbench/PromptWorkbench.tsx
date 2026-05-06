import { Badge, Box, Button, Flex, IconButton, Text, Tooltip } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { adjustPromptAttention } from 'common/util/promptAttention';
import { selectModel } from 'features/controlLayers/store/paramsSlice';
import {
  modeChanged,
  randomRefreshModeChanged,
  selectDynamicPromptsIsLoading,
  selectDynamicPromptsMode,
  selectDynamicPromptsParsingError,
  selectDynamicPromptsPrompts,
  selectDynamicPromptsRandomRefreshMode,
} from 'features/dynamicPrompts/store/dynamicPromptsSlice';
import { selectSystemPrefersNumericAttentionWeights } from 'features/system/store/systemSlice';
import type { MouseEvent, RefObject } from 'react';
import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { flushSync } from 'react-dom';
import { PiDiceFiveBold, PiPushPinSimpleBold, PiRepeatBold, PiSquaresFourBold } from 'react-icons/pi';
import type { WildcardIndexItem } from 'services/api/endpoints/utilities';
import { useLazyWildcardValuesQuery, useWildcardsQuery } from 'services/api/endpoints/utilities';

import { getPromptDiagnostics, type PromptDiagnosticSeverity } from './diagnostics';
import { getPromptModelCapabilities } from './modelCapabilities';
import {
  getPromptWorkbenchOccurrences,
  type PromptRange,
  type PromptWeightOccurrence,
  type PromptWildcardOccurrence,
  removePromptRange,
  replacePromptRange,
} from './occurrences';
import { PromptInspector } from './PromptInspector';
import {
  applyWildcardCompletion,
  filterWildcardOptions,
  getCyclicWildcardToken,
  getWildcardAutocompleteStatusMessage,
  getWildcardCompletionContext,
  getWildcardDisplayPath,
  type WildcardCompletionContext,
} from './wildcards';

type PromptWorkbenchProps = {
  prompt: string;
  textareaRef: RefObject<HTMLTextAreaElement | null>;
  onPromptChange: (prompt: string) => void;
};

type SelectionRange = {
  start: number;
  end: number;
};

const EMPTY_WILDCARDS: [] = [];
const WILDCARD_ACTION_BUTTON_SIZE = 7;

export const PromptWorkbench = memo(({ prompt, textareaRef, onPromptChange }: PromptWorkbenchProps) => {
  const dispatch = useAppDispatch();
  const model = useAppSelector(selectModel);
  const prefersNumericWeights = useAppSelector(selectSystemPrefersNumericAttentionWeights);
  const dynamicPromptMode = useAppSelector(selectDynamicPromptsMode);
  const dynamicPromptRandomRefreshMode = useAppSelector(selectDynamicPromptsRandomRefreshMode);
  const dynamicPrompts = useAppSelector(selectDynamicPromptsPrompts);
  const dynamicPromptError = useAppSelector(selectDynamicPromptsParsingError);
  const isDynamicPromptsLoading = useAppSelector(selectDynamicPromptsIsLoading);
  const { data: wildcardsData, isError: isWildcardIndexUnavailable, isFetching: isFetchingWildcards } = useWildcardsQuery();
  const [loadWildcardValues, wildcardValuesResult] = useLazyWildcardValuesQuery();
  const wildcards = wildcardsData?.wildcards ?? EMPTY_WILDCARDS;
  const wildcardIndexErrorCount = wildcardsData?.errors.length ?? 0;
  const capabilities = useMemo(() => getPromptModelCapabilities(model?.base), [model?.base]);
  const [caret, setCaret] = useState(0);
  const [isFocused, setIsFocused] = useState(false);
  const [fixedWildcardPath, setFixedWildcardPath] = useState<string | null>(null);
  const [fixedWildcardContext, setFixedWildcardContext] = useState<WildcardCompletionContext | null>(null);
  const [fixedWildcardOccurrence, setFixedWildcardOccurrence] = useState<PromptWildcardOccurrence | null>(null);
  const selectionRef = useRef<SelectionRange>({ start: 0, end: 0 });

  const diagnostics = useMemo(
    () =>
      getPromptDiagnostics({
        prompt,
        modelBase: model?.base,
        wildcards,
        wildcardIndexUnavailable: isWildcardIndexUnavailable,
        wildcardIndexErrorCount,
        dynamicPromptCount: dynamicPrompts.length,
        dynamicPromptMode,
        dynamicPromptRandomRefreshMode,
        dynamicPromptError,
      }),
    [
      dynamicPromptError,
      dynamicPromptMode,
      dynamicPromptRandomRefreshMode,
      dynamicPrompts.length,
      isWildcardIndexUnavailable,
      model?.base,
      prompt,
      wildcardIndexErrorCount,
      wildcards,
    ]
  );

  const completionContext = useMemo(
    () => (isFocused ? getWildcardCompletionContext(prompt, caret) : null),
    [caret, isFocused, prompt]
  );

  const wildcardOptions = useMemo(() => {
    if (!completionContext) {
      return [];
    }
    return filterWildcardOptions(wildcards, completionContext.query);
  }, [completionContext, wildcards]);

  const promptWorkbenchOccurrences = useMemo(
    () =>
      getPromptWorkbenchOccurrences({
        prompt,
        wildcards,
        wildcardIndexUnavailable: isWildcardIndexUnavailable,
        dynamicPromptMode,
        supportsAttentionWeights: capabilities.supportsAttentionWeights,
      }),
    [capabilities.supportsAttentionWeights, dynamicPromptMode, isWildcardIndexUnavailable, prompt, wildcards]
  );

  const wildcardStatusMessage = useMemo(() => {
    if (!completionContext) {
      return null;
    }

    return getWildcardAutocompleteStatusMessage({
      isLoading: isFetchingWildcards,
      isUnavailable: isWildcardIndexUnavailable,
      optionCount: wildcardOptions.length,
      query: completionContext.query,
      wildcardCount: wildcards.length,
    });
  }, [completionContext, isFetchingWildcards, isWildcardIndexUnavailable, wildcardOptions.length, wildcards.length]);

  const fixedWildcardValues =
    wildcardValuesResult.currentData?.path === fixedWildcardPath ? wildcardValuesResult.currentData.values : null;

  const activeFixedWildcardOccurrenceId = fixedWildcardOccurrence?.id ?? null;

  const syncSelection = useCallback(() => {
    const textarea = textareaRef.current;
    if (!textarea) {
      return;
    }
    const start = textarea.selectionStart ?? 0;
    const end = textarea.selectionEnd ?? start;
    selectionRef.current = { start, end };
    setCaret(end);
    setIsFocused(document.activeElement === textarea);
  }, [textareaRef]);

  useEffect(() => {
    const textarea = textareaRef.current;
    if (!textarea) {
      return;
    }
    let blurTimeout: number | undefined;
    const syncBlurred = () => {
      blurTimeout = window.setTimeout(() => {
        setIsFocused(false);
      }, 150);
    };

    textarea.addEventListener('keyup', syncSelection);
    textarea.addEventListener('click', syncSelection);
    textarea.addEventListener('select', syncSelection);
    textarea.addEventListener('focus', syncSelection);
    textarea.addEventListener('blur', syncBlurred);

    return () => {
      textarea.removeEventListener('keyup', syncSelection);
      textarea.removeEventListener('click', syncSelection);
      textarea.removeEventListener('select', syncSelection);
      textarea.removeEventListener('focus', syncSelection);
      textarea.removeEventListener('blur', syncBlurred);
      if (blurTimeout !== undefined) {
        window.clearTimeout(blurTimeout);
      }
    };
  }, [syncSelection, textareaRef]);

  useEffect(() => {
    if (!completionContext) {
      setFixedWildcardContext(null);
      if (!fixedWildcardOccurrence) {
        setFixedWildcardPath(null);
      }
    }
  }, [completionContext, fixedWildcardOccurrence]);

  useEffect(() => {
    if (
      fixedWildcardOccurrence &&
      !promptWorkbenchOccurrences.some((occurrence) => occurrence.id === fixedWildcardOccurrence.id)
    ) {
      setFixedWildcardOccurrence(null);
    }
  }, [fixedWildcardOccurrence, promptWorkbenchOccurrences]);

  const focusPromptRange = useCallback(
    (range: PromptRange) => {
      requestAnimationFrame(() => {
        const textarea = textareaRef.current;
        textarea?.focus();
        textarea?.setSelectionRange(range.start, range.end);
        selectionRef.current = { start: range.start, end: range.end };
        setCaret(range.end);
        setIsFocused(document.activeElement === textarea);
      });
    },
    [textareaRef]
  );

  const applyPromptReplacement = useCallback(
    (nextPrompt: string, selection: SelectionRange) => {
      flushSync(() => {
        onPromptChange(nextPrompt);
      });

      requestAnimationFrame(() => {
        const textarea = textareaRef.current;
        textarea?.focus();
        textarea?.setSelectionRange(selection.start, selection.end);
        selectionRef.current = selection;
        setCaret(selection.end);
        setFixedWildcardPath(null);
        setFixedWildcardContext(null);
        setFixedWildcardOccurrence(null);
      });
    },
    [onPromptChange, textareaRef]
  );

  const replaceOccurrenceRange = useCallback(
    (range: PromptRange, replacement: string) => {
      const result = replacePromptRange(prompt, range, replacement);
      applyPromptReplacement(result.prompt, { start: result.caret, end: result.caret });
    },
    [applyPromptReplacement, prompt]
  );

  const replaceCompletion = useCallback(
    (replacement: string, context: WildcardCompletionContext) => {
      const result = applyWildcardCompletion(prompt, context, replacement);

      flushSync(() => {
        onPromptChange(result.prompt);
      });

      requestAnimationFrame(() => {
        const textarea = textareaRef.current;
        textarea?.focus();
        textarea?.setSelectionRange(result.caret, result.caret);
        selectionRef.current = { start: result.caret, end: result.caret };
        setCaret(result.caret);
        setFixedWildcardPath(null);
        setFixedWildcardContext(null);
        setFixedWildcardOccurrence(null);
      });
    },
    [onPromptChange, prompt, textareaRef]
  );

  const onWildcardMouseDown = useCallback(
    (token: string) => (e: MouseEvent<HTMLButtonElement>) => {
      e.preventDefault();
      if (!completionContext) {
        return;
      }
      dispatch(modeChanged('random'));
      dispatch(randomRefreshModeChanged('per_enqueue'));
      replaceCompletion(token, completionContext);
    },
    [completionContext, dispatch, replaceCompletion]
  );

  const onRandomWildcardMouseDown = useCallback(
    (wildcard: WildcardIndexItem) => (e: MouseEvent<HTMLButtonElement>) => {
      e.preventDefault();
      if (!completionContext) {
        return;
      }
      dispatch(modeChanged('random'));
      dispatch(randomRefreshModeChanged('per_enqueue'));
      replaceCompletion(wildcard.token, completionContext);
    },
    [completionContext, dispatch, replaceCompletion]
  );

  const onCyclicWildcardMouseDown = useCallback(
    (wildcard: WildcardIndexItem) => (e: MouseEvent<HTMLButtonElement>) => {
      e.preventDefault();
      if (!completionContext) {
        return;
      }
      dispatch(modeChanged('random'));
      dispatch(randomRefreshModeChanged('manual'));
      replaceCompletion(getCyclicWildcardToken(wildcard.path), completionContext);
    },
    [completionContext, dispatch, replaceCompletion]
  );

  const onExploreAllMouseDown = useCallback(
    (wildcard: WildcardIndexItem) => (e: MouseEvent<HTMLButtonElement>) => {
      e.preventDefault();
      if (!completionContext) {
        return;
      }
      dispatch(modeChanged('combinatorial'));
      dispatch(randomRefreshModeChanged('manual'));
      replaceCompletion(wildcard.token, completionContext);
    },
    [completionContext, dispatch, replaceCompletion]
  );

  const onPickFixedMouseDown = useCallback(
    (wildcard: WildcardIndexItem) => (e: MouseEvent<HTMLButtonElement>) => {
      e.preventDefault();
      if (!completionContext) {
        return;
      }
      setFixedWildcardPath(wildcard.path);
      setFixedWildcardContext(completionContext);
      setFixedWildcardOccurrence(null);
      loadWildcardValues({ path: wildcard.path, limit: 200 });
    },
    [completionContext, loadWildcardValues]
  );

  const onFixedValueMouseDown = useCallback(
    (value: string) => (e: MouseEvent<HTMLButtonElement>) => {
      e.preventDefault();
      if (fixedWildcardContext) {
        replaceCompletion(value, fixedWildcardContext);
        return;
      }
      if (fixedWildcardOccurrence) {
        replaceOccurrenceRange(fixedWildcardOccurrence.range, value);
      }
    },
    [fixedWildcardContext, fixedWildcardOccurrence, replaceCompletion, replaceOccurrenceRange]
  );

  const onInspectorFixedValue = useCallback(
    (value: string) => {
      if (!fixedWildcardOccurrence) {
        return;
      }
      replaceOccurrenceRange(fixedWildcardOccurrence.range, value);
    },
    [fixedWildcardOccurrence, replaceOccurrenceRange]
  );

  const adjustWeight = useCallback(
    (direction: 'increment' | 'decrement') => {
      const textarea = textareaRef.current;
      const selection = selectionRef.current;
      const result = adjustPromptAttention(prompt, selection.start, selection.end, direction, prefersNumericWeights);

      flushSync(() => {
        onPromptChange(result.prompt);
      });

      requestAnimationFrame(() => {
        textarea?.focus();
        textarea?.setSelectionRange(result.selectionStart, result.selectionEnd);
        selectionRef.current = { start: result.selectionStart, end: result.selectionEnd };
        setCaret(result.selectionEnd);
      });
    },
    [onPromptChange, prefersNumericWeights, prompt, textareaRef]
  );

  const onDecrementMouseDown = useCallback((e: MouseEvent<HTMLButtonElement>) => {
    e.preventDefault();
  }, []);

  const onIncrementMouseDown = useCallback((e: MouseEvent<HTMLButtonElement>) => {
    e.preventDefault();
  }, []);

  const onDecrementClick = useCallback(() => {
    adjustWeight('decrement');
  }, [adjustWeight]);

  const onIncrementClick = useCallback(() => {
    adjustWeight('increment');
  }, [adjustWeight]);

  const onRemoveWildcardOccurrence = useCallback(
    (occurrence: PromptWildcardOccurrence) => {
      const result = removePromptRange(prompt, occurrence.range);
      applyPromptReplacement(result.prompt, { start: result.caret, end: result.caret });
    },
    [applyPromptReplacement, prompt]
  );

  const onRandomWildcardOccurrence = useCallback(
    (occurrence: PromptWildcardOccurrence) => {
      dispatch(modeChanged('random'));
      dispatch(randomRefreshModeChanged('per_enqueue'));
      replaceOccurrenceRange(occurrence.range, `__${occurrence.path}__`);
    },
    [dispatch, replaceOccurrenceRange]
  );

  const onPickFixedWildcardOccurrence = useCallback(
    (occurrence: PromptWildcardOccurrence) => {
      if (!occurrence.wildcard) {
        return;
      }
      setFixedWildcardPath(occurrence.path);
      setFixedWildcardContext(null);
      setFixedWildcardOccurrence(occurrence);
      loadWildcardValues({ path: occurrence.path, limit: 200 });
    },
    [loadWildcardValues]
  );

  const onCyclicWildcardOccurrence = useCallback(
    (occurrence: PromptWildcardOccurrence) => {
      dispatch(modeChanged('random'));
      dispatch(randomRefreshModeChanged('manual'));
      replaceOccurrenceRange(occurrence.range, getCyclicWildcardToken(occurrence.path));
    },
    [dispatch, replaceOccurrenceRange]
  );

  const onExploreAllWildcardOccurrence = useCallback(
    (occurrence: PromptWildcardOccurrence) => {
      dispatch(modeChanged('combinatorial'));
      dispatch(randomRefreshModeChanged('manual'));
      replaceOccurrenceRange(occurrence.range, `__${occurrence.path}__`);
    },
    [dispatch, replaceOccurrenceRange]
  );

  const onAdjustWeightOccurrence = useCallback(
    (occurrence: PromptWeightOccurrence, direction: 'increment' | 'decrement') => {
      const result = adjustPromptAttention(
        prompt,
        occurrence.range.start,
        occurrence.range.end,
        direction,
        prefersNumericWeights
      );
      applyPromptReplacement(result.prompt, { start: result.selectionStart, end: result.selectionEnd });
    },
    [applyPromptReplacement, prefersNumericWeights, prompt]
  );

  return (
    <Flex flexDir="column" gap={1} mt={1}>
      <Flex gap={1} alignItems="center" flexWrap="wrap">
        {diagnostics.map((diagnostic) => (
          <Tooltip key={diagnostic.code} label={diagnostic.description}>
            <Badge size="sm" colorScheme={getDiagnosticColorScheme(diagnostic.severity)}>
              {diagnostic.label}
            </Badge>
          </Tooltip>
        ))}
        {isDynamicPromptsLoading && (
          <Badge size="sm" colorScheme="blue">
            Dynamic loading
          </Badge>
        )}
        {isFetchingWildcards && (
          <Badge size="sm" colorScheme="blue">
            Wildcards loading
          </Badge>
        )}
        <Flex gap={1} ms="auto">
          <Tooltip label={capabilities.attentionWeightsLabel}>
            <Button
              size="xs"
              variant="outline"
              isDisabled={!capabilities.supportsAttentionWeights}
              onMouseDown={onDecrementMouseDown}
              onClick={onDecrementClick}
            >
              -
            </Button>
          </Tooltip>
          <Tooltip label={capabilities.attentionWeightsLabel}>
            <Button
              size="xs"
              variant="outline"
              isDisabled={!capabilities.supportsAttentionWeights}
              onMouseDown={onIncrementMouseDown}
              onClick={onIncrementClick}
            >
              +
            </Button>
          </Tooltip>
        </Flex>
      </Flex>
      {completionContext && (wildcardOptions.length > 0 || wildcardStatusMessage) && (
        <Box>
          <Flex
            flexDir="column"
            borderWidth={1}
            borderColor="base.700"
            borderRadius="base"
            overflow="hidden"
            maxH={32}
            overflowY="auto"
            data-testid="prompt-wildcard-autocomplete"
          >
            {wildcardStatusMessage && (
              <Flex alignItems="center" minH={8} px={3}>
                <Text fontSize="xs" color={isWildcardIndexUnavailable ? 'error.300' : 'base.400'} noOfLines={2}>
                  {wildcardStatusMessage}
                </Text>
              </Flex>
            )}
            {wildcardOptions.map((wildcard) => (
              <Box key={wildcard.path}>
                <Flex
                  alignItems="center"
                  gap={1}
                  px={1}
                  minH={8}
                  _hover={{ bg: 'base.800' }}
                  transitionProperty="common"
                  transitionDuration="0.1s"
                >
                  <Tooltip label={wildcard.samples.join(', ')}>
                    <Button
                      size="xs"
                      variant="ghost"
                      justifyContent="flex-start"
                      borderRadius="base"
                      flexGrow={1}
                      minW={0}
                      h={7}
                      px={2}
                      onMouseDown={onWildcardMouseDown(wildcard.token)}
                    >
                      <Text as="span" noOfLines={1} color="base.100" fontWeight="semibold">
                        {getWildcardDisplayPath(wildcard)}
                      </Text>
                    </Button>
                  </Tooltip>
                  <Text
                    fontSize="sm"
                    fontFamily="mono"
                    color="base.400"
                    textAlign="end"
                    w={8}
                    flexShrink={0}
                  >
                    {wildcard.value_count}
                  </Text>
                  <Flex gap={0.5} flexShrink={0}>
                    <IconButton
                      aria-label={`Insert ${wildcard.path} as random wildcard`}
                      tooltip="Random every Invoke"
                      size="xs"
                      variant="ghost"
                      minW={WILDCARD_ACTION_BUTTON_SIZE}
                      h={WILDCARD_ACTION_BUTTON_SIZE}
                      icon={<PiDiceFiveBold />}
                      onMouseDown={onRandomWildcardMouseDown(wildcard)}
                    />
                    <IconButton
                      aria-label={`Pick a fixed value from ${wildcard.path}`}
                      tooltip="Pick fixed value"
                      size="xs"
                      variant="ghost"
                      minW={WILDCARD_ACTION_BUTTON_SIZE}
                      h={WILDCARD_ACTION_BUTTON_SIZE}
                      icon={<PiPushPinSimpleBold />}
                      onMouseDown={onPickFixedMouseDown(wildcard)}
                    />
                    <IconButton
                      aria-label={`Insert ${wildcard.path} as cyclic wildcard`}
                      tooltip="Cycle"
                      size="xs"
                      variant="ghost"
                      minW={WILDCARD_ACTION_BUTTON_SIZE}
                      h={WILDCARD_ACTION_BUTTON_SIZE}
                      icon={<PiRepeatBold />}
                      onMouseDown={onCyclicWildcardMouseDown(wildcard)}
                    />
                    <IconButton
                      aria-label={`Explore all combinations for ${wildcard.path}`}
                      tooltip="Explore all"
                      size="xs"
                      variant="ghost"
                      minW={WILDCARD_ACTION_BUTTON_SIZE}
                      h={WILDCARD_ACTION_BUTTON_SIZE}
                      icon={<PiSquaresFourBold />}
                      onMouseDown={onExploreAllMouseDown(wildcard)}
                    />
                  </Flex>
                </Flex>
                {fixedWildcardPath === wildcard.path && (
                  <Flex flexDir="column" ps={4} pe={1} pb={1} gap={0.5}>
                    {wildcardValuesResult.isFetching && (
                      <Text fontSize="xs" color="base.400">
                        Loading values...
                      </Text>
                    )}
                    {fixedWildcardValues?.map((value) => (
                      <Button
                        key={value}
                        size="xs"
                        variant="ghost"
                        justifyContent="flex-start"
                        h={7}
                        px={2}
                        onMouseDown={onFixedValueMouseDown(value)}
                      >
                        <Text as="span" noOfLines={1}>
                          {value}
                        </Text>
                      </Button>
                    ))}
                  </Flex>
                )}
              </Box>
            ))}
          </Flex>
          <Text mt={1} fontSize="xs" color="base.400">
            {completionContext.query ? `Matching "${completionContext.query}"` : 'Local wildcards'}
          </Text>
        </Box>
      )}
      <PromptInspector
        occurrences={promptWorkbenchOccurrences}
        randomRefreshMode={dynamicPromptRandomRefreshMode}
        fixedWildcardOccurrenceId={activeFixedWildcardOccurrenceId}
        fixedWildcardValues={fixedWildcardValues}
        isFetchingFixedWildcardValues={wildcardValuesResult.isFetching}
        onSelectRange={focusPromptRange}
        onRemoveWildcard={onRemoveWildcardOccurrence}
        onRandomWildcard={onRandomWildcardOccurrence}
        onPickFixedWildcard={onPickFixedWildcardOccurrence}
        onCyclicWildcard={onCyclicWildcardOccurrence}
        onExploreAllWildcard={onExploreAllWildcardOccurrence}
        onFixedValue={onInspectorFixedValue}
        onAdjustWeight={onAdjustWeightOccurrence}
      />
    </Flex>
  );
});

PromptWorkbench.displayName = 'PromptWorkbench';

const getDiagnosticColorScheme = (severity: PromptDiagnosticSeverity): string => {
  switch (severity) {
    case 'ok':
      return 'green';
    case 'warning':
      return 'yellow';
    case 'error':
      return 'red';
    case 'info':
      return 'base';
  }
};
