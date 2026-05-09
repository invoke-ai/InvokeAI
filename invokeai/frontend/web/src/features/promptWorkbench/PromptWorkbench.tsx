import { Box, Button, Flex, Menu, MenuButton, MenuItem, MenuList, Text, Tooltip } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { adjustPromptAttention } from 'common/util/promptAttention';
import { selectModel } from 'features/controlLayers/store/paramsSlice';
import { useDynamicPromptsModal } from 'features/dynamicPrompts/hooks/useDynamicPromptsModal';
import {
  modeChanged,
  randomRefreshModeChanged,
  selectDynamicPromptsMode,
  selectDynamicPromptsParsingError,
  selectDynamicPromptsPrompts,
  selectDynamicPromptsRandomRefreshMode,
} from 'features/dynamicPrompts/store/dynamicPromptsSlice';
import { selectSystemPrefersNumericAttentionWeights } from 'features/system/store/systemSlice';
import type { MouseEvent, RefObject } from 'react';
import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { flushSync } from 'react-dom';
import { useTranslation } from 'react-i18next';
import { PiCubeBold, PiDiceFiveBold } from 'react-icons/pi';
import type { WildcardIndexItem } from 'services/api/endpoints/utilities';
import { useLazyWildcardValuesQuery, useWildcardsQuery } from 'services/api/endpoints/utilities';

import { getPromptDiagnostics, type PromptDiagnosticSeverity } from './diagnostics';
import type { PromptWorkbenchTranslation } from './i18n';
import { clampNavigationIndex, getNextNavigationIndex, getPromptWorkbenchKeyboardIntent } from './keyboardNavigation';
import { getPromptModelCapabilities } from './modelCapabilities';
import {
  getPromptWorkbenchOccurrences,
  getWildcardBehaviorActionIntent,
  type PromptRange,
  type PromptWeightOccurrence,
  type PromptWildcardOccurrence,
  removePromptRange,
  replacePromptRange,
  type WildcardBehaviorAction,
} from './occurrences';
import { PromptInspector } from './PromptInspector';
import { PromptWildcardBehaviorMenu } from './PromptWildcardBehaviorMenu';
import { PromptWorkbenchBadge, type PromptWorkbenchBadgeTone } from './PromptWorkbenchBadge';
import {
  applyWildcardCompletion,
  filterWildcardOptions,
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
const getCompletionContextKey = (context: WildcardCompletionContext): string =>
  `${context.start}:${context.end}:${context.query}`;
const PROMPT_INTENT_PANEL_BG = 'linear-gradient(180deg, rgba(15, 23, 31, 0.92) 0%, rgba(11, 18, 25, 0.96) 100%)';
const PROMPT_INTENT_PANEL_BORDER = 'rgba(126, 143, 164, 0.28)';

export const PromptWorkbench = memo(({ prompt, textareaRef, onPromptChange }: PromptWorkbenchProps) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const { onOpen: onOpenDynamicPromptsModal } = useDynamicPromptsModal();
  const model = useAppSelector(selectModel);
  const prefersNumericWeights = useAppSelector(selectSystemPrefersNumericAttentionWeights);
  const dynamicPromptMode = useAppSelector(selectDynamicPromptsMode);
  const dynamicPromptRandomRefreshMode = useAppSelector(selectDynamicPromptsRandomRefreshMode);
  const dynamicPrompts = useAppSelector(selectDynamicPromptsPrompts);
  const dynamicPromptError = useAppSelector(selectDynamicPromptsParsingError);
  const {
    data: wildcardsData,
    isError: isWildcardIndexUnavailable,
    isFetching: isFetchingWildcards,
  } = useWildcardsQuery();
  const [loadWildcardValues, wildcardValuesResult] = useLazyWildcardValuesQuery();
  const wildcards = wildcardsData?.wildcards ?? EMPTY_WILDCARDS;
  const wildcardIndexErrorCount = wildcardsData?.errors.length ?? 0;
  const capabilities = useMemo(() => getPromptModelCapabilities(model?.base), [model?.base]);
  const [caret, setCaret] = useState(0);
  const [isFocused, setIsFocused] = useState(false);
  const [fixedWildcardPath, setFixedWildcardPath] = useState<string | null>(null);
  const [fixedWildcardContext, setFixedWildcardContext] = useState<WildcardCompletionContext | null>(null);
  const [fixedWildcardOccurrence, setFixedWildcardOccurrence] = useState<PromptWildcardOccurrence | null>(null);
  const [isAutocompleteBehaviorMenuOpen, setIsAutocompleteBehaviorMenuOpen] = useState(false);
  const [dismissedCompletionContextKey, setDismissedCompletionContextKey] = useState<string | null>(null);
  const [activeWildcardIndex, setActiveWildcardIndex] = useState(0);
  const [activeFixedValueIndex, setActiveFixedValueIndex] = useState(0);
  const selectionRef = useRef<SelectionRange>({ start: 0, end: 0 });
  const [selection, setSelection] = useState<SelectionRange>({ start: 0, end: 0 });
  const completionContextRef = useRef<WildcardCompletionContext | null>(null);
  const wildcardOptionElementsRef = useRef<Array<HTMLElement | null>>([]);
  const fixedValueElementsRef = useRef<Array<HTMLElement | null>>([]);
  const translate = useCallback(
    (translation: PromptWorkbenchTranslation) => t(translation.key, translation.options),
    [t]
  );

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

  const rawCompletionContextBase = useMemo(
    () => (isFocused ? getWildcardCompletionContext(prompt, caret) : null),
    [caret, isFocused, prompt]
  );

  const rawCompletionContext = useMemo(() => {
    if (!rawCompletionContextBase) {
      return null;
    }

    return getCompletionContextKey(rawCompletionContextBase) === dismissedCompletionContextKey
      ? null
      : rawCompletionContextBase;
  }, [dismissedCompletionContextKey, rawCompletionContextBase]);

  useEffect(() => {
    if (rawCompletionContext) {
      completionContextRef.current = rawCompletionContext;
    }
  }, [rawCompletionContext]);

  const completionContext = useMemo(
    () =>
      rawCompletionContext ??
      fixedWildcardContext ??
      (isAutocompleteBehaviorMenuOpen ? completionContextRef.current : null),
    [fixedWildcardContext, isAutocompleteBehaviorMenuOpen, rawCompletionContext]
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
  const hasFixedValuePicker = Boolean(fixedWildcardPath && (fixedWildcardValues || wildcardValuesResult.isFetching));
  const fixedValueCount = fixedWildcardValues?.length ?? 0;

  const setWildcardOptionElement = useCallback((index: number, element: HTMLElement | null) => {
    wildcardOptionElementsRef.current[index] = element;
  }, []);

  const setFixedValueElement = useCallback((index: number, element: HTMLElement | null) => {
    fixedValueElementsRef.current[index] = element;
  }, []);

  const wildcardOptionElementSetters = useMemo(
    () =>
      wildcardOptions.map((_wildcard, index) => (element: HTMLElement | null) => {
        setWildcardOptionElement(index, element);
      }),
    [setWildcardOptionElement, wildcardOptions]
  );

  const fixedValueElementSetters = useMemo(
    () =>
      (fixedWildcardValues ?? []).map((_value, index) => (element: HTMLElement | null) => {
        setFixedValueElement(index, element);
      }),
    [fixedWildcardValues, setFixedValueElement]
  );

  useEffect(() => {
    setActiveWildcardIndex((currentIndex) => clampNavigationIndex(currentIndex, wildcardOptions.length));
    wildcardOptionElementsRef.current = wildcardOptionElementsRef.current.slice(0, wildcardOptions.length);
  }, [wildcardOptions.length]);

  useEffect(() => {
    setActiveWildcardIndex(0);
  }, [completionContext?.query, completionContext?.start]);

  useEffect(() => {
    setActiveFixedValueIndex((currentIndex) => clampNavigationIndex(currentIndex, fixedValueCount));
    fixedValueElementsRef.current = fixedValueElementsRef.current.slice(0, fixedValueCount);
  }, [fixedValueCount]);

  useEffect(() => {
    setActiveFixedValueIndex(0);
  }, [fixedWildcardPath]);

  useEffect(() => {
    wildcardOptionElementsRef.current[activeWildcardIndex]?.scrollIntoView({ block: 'nearest' });
  }, [activeWildcardIndex]);

  useEffect(() => {
    fixedValueElementsRef.current[activeFixedValueIndex]?.scrollIntoView({ block: 'nearest' });
  }, [activeFixedValueIndex]);

  const syncSelection = useCallback(() => {
    const textarea = textareaRef.current;
    if (!textarea) {
      return;
    }
    const isTextareaFocused = document.activeElement === textarea;
    if (!isTextareaFocused) {
      setSelection({ start: 0, end: 0 });
      setIsFocused(false);
      return;
    }
    const start = textarea.selectionStart ?? 0;
    const end = textarea.selectionEnd ?? start;
    selectionRef.current = { start, end };
    setSelection({ start, end });
    setCaret(end);
    setIsFocused(true);
  }, [textareaRef]);

  useEffect(() => {
    const textarea = textareaRef.current;
    if (!textarea) {
      return;
    }
    let blurTimeout: number | undefined;
    let selectionFrame: number | undefined;
    const syncSelectionSoon = () => {
      selectionFrame = window.requestAnimationFrame(syncSelection);
    };
    const syncBlurred = () => {
      blurTimeout = window.setTimeout(() => {
        setSelection({ start: 0, end: 0 });
        setIsFocused(false);
      }, 150);
    };

    textarea.addEventListener('keyup', syncSelectionSoon);
    textarea.addEventListener('click', syncSelectionSoon);
    textarea.addEventListener('mouseup', syncSelectionSoon);
    textarea.addEventListener('input', syncSelectionSoon);
    textarea.addEventListener('select', syncSelectionSoon);
    textarea.addEventListener('focus', syncSelectionSoon);
    textarea.addEventListener('blur', syncBlurred);
    document.addEventListener('selectionchange', syncSelectionSoon);

    return () => {
      textarea.removeEventListener('keyup', syncSelectionSoon);
      textarea.removeEventListener('click', syncSelectionSoon);
      textarea.removeEventListener('mouseup', syncSelectionSoon);
      textarea.removeEventListener('input', syncSelectionSoon);
      textarea.removeEventListener('select', syncSelectionSoon);
      textarea.removeEventListener('focus', syncSelectionSoon);
      textarea.removeEventListener('blur', syncBlurred);
      document.removeEventListener('selectionchange', syncSelectionSoon);
      if (blurTimeout !== undefined) {
        window.clearTimeout(blurTimeout);
      }
      if (selectionFrame !== undefined) {
        window.cancelAnimationFrame(selectionFrame);
      }
    };
  }, [syncSelection, textareaRef]);

  useEffect(() => {
    if (!completionContext && !isAutocompleteBehaviorMenuOpen && !fixedWildcardContext) {
      setFixedWildcardContext(null);
      if (!fixedWildcardOccurrence) {
        setFixedWildcardPath(null);
      }
    }
  }, [completionContext, fixedWildcardContext, fixedWildcardOccurrence, isAutocompleteBehaviorMenuOpen]);

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
        setSelection({ start: range.start, end: range.end });
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
        setSelection(selection);
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
        setSelection({ start: result.caret, end: result.caret });
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
      dispatch(randomRefreshModeChanged('per_image'));
      replaceCompletion(token, completionContext);
    },
    [completionContext, dispatch, replaceCompletion]
  );

  const openFixedValuesForAutocompleteWildcard = useCallback(
    (wildcard: WildcardIndexItem, context: WildcardCompletionContext) => {
      setFixedWildcardPath(wildcard.path);
      setFixedWildcardContext(context);
      setFixedWildcardOccurrence(null);
      setActiveFixedValueIndex(0);
      loadWildcardValues({ path: wildcard.path, limit: 200 });
    },
    [loadWildcardValues]
  );

  const onAutocompleteWildcardBehaviorAction = useCallback(
    (wildcard: WildcardIndexItem) => (action: WildcardBehaviorAction) => {
      const context = completionContext ?? completionContextRef.current;
      if (!context) {
        return;
      }

      const intent = getWildcardBehaviorActionIntent(action, wildcard.path);
      if (intent.opensFixedValues) {
        openFixedValuesForAutocompleteWildcard(wildcard, context);
        return;
      }

      if (intent.replacement) {
        replaceCompletion(intent.replacement, context);
      }
    },
    [completionContext, openFixedValuesForAutocompleteWildcard, replaceCompletion]
  );

  const onAutocompleteBehaviorMenuOpen = useCallback(() => {
    setIsAutocompleteBehaviorMenuOpen(true);
  }, []);

  const onAutocompleteBehaviorMenuClose = useCallback(() => {
    setIsAutocompleteBehaviorMenuOpen(false);
  }, []);

  const applyFixedValue = useCallback(
    (value: string) => {
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

  const onFixedValueMouseDown = useCallback(
    (value: string) => (e: MouseEvent<HTMLButtonElement>) => {
      e.preventDefault();
      applyFixedValue(value);
    },
    [applyFixedValue]
  );

  const onInspectorFixedValue = useCallback(
    (value: string) => {
      applyFixedValue(value);
    },
    [applyFixedValue]
  );

  const dismissCompletion = useCallback(() => {
    const context = fixedWildcardContext ?? rawCompletionContextBase ?? completionContextRef.current;
    if (context) {
      setDismissedCompletionContextKey(getCompletionContextKey(context));
    }
    setFixedWildcardPath(null);
    setFixedWildcardContext(null);
    setFixedWildcardOccurrence(null);
    setIsAutocompleteBehaviorMenuOpen(false);

    requestAnimationFrame(() => {
      textareaRef.current?.focus();
    });
  }, [fixedWildcardContext, rawCompletionContextBase, textareaRef]);

  const onPromptWorkbenchKeyDown = useCallback(
    (e: KeyboardEvent) => {
      const keyboardTarget = hasFixedValuePicker
        ? 'fixed_values'
        : completionContext && wildcardOptions.length > 0
          ? 'autocomplete'
          : null;

      if (!keyboardTarget) {
        return;
      }

      const intent = getPromptWorkbenchKeyboardIntent({
        key: e.key,
        shiftKey: e.shiftKey,
        target: keyboardTarget,
      });

      if (!intent) {
        return;
      }

      e.preventDefault();
      e.stopPropagation();

      if (intent === 'dismiss') {
        dismissCompletion();
        return;
      }

      if (intent === 'next' || intent === 'previous') {
        if (keyboardTarget === 'fixed_values') {
          setActiveFixedValueIndex((currentIndex) =>
            getNextNavigationIndex({
              currentIndex,
              direction: intent,
              itemCount: fixedValueCount,
            })
          );
          return;
        }

        setActiveWildcardIndex((currentIndex) =>
          getNextNavigationIndex({
            currentIndex,
            direction: intent,
            itemCount: wildcardOptions.length,
          })
        );
        return;
      }

      if (intent === 'insert_fixed_value') {
        const value = fixedWildcardValues?.[activeFixedValueIndex];
        if (value) {
          applyFixedValue(value);
        }
        return;
      }

      const wildcard = wildcardOptions[activeWildcardIndex];
      if (!wildcard || !completionContext) {
        return;
      }

      if (intent === 'open_fixed_values') {
        openFixedValuesForAutocompleteWildcard(wildcard, completionContext);
        return;
      }

      if (intent === 'insert_wildcard') {
        dispatch(modeChanged('random'));
        dispatch(randomRefreshModeChanged('per_image'));
        replaceCompletion(wildcard.token, completionContext);
      }
    },
    [
      activeFixedValueIndex,
      activeWildcardIndex,
      applyFixedValue,
      completionContext,
      dismissCompletion,
      dispatch,
      fixedValueCount,
      fixedWildcardValues,
      hasFixedValuePicker,
      openFixedValuesForAutocompleteWildcard,
      replaceCompletion,
      wildcardOptions,
    ]
  );

  useEffect(() => {
    const textarea = textareaRef.current;
    if (!textarea) {
      return;
    }

    textarea.addEventListener('keydown', onPromptWorkbenchKeyDown);

    return () => {
      textarea.removeEventListener('keydown', onPromptWorkbenchKeyDown);
    };
  }, [onPromptWorkbenchKeyDown, textareaRef]);

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
        setSelection({ start: result.selectionStart, end: result.selectionEnd });
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

  const onOpenQueuedOutputsMouseDown = useCallback((e: MouseEvent<HTMLButtonElement>) => {
    e.preventDefault();
  }, []);

  const onOpenQueuedOutputsClick = useCallback(() => {
    onOpenDynamicPromptsModal();
  }, [onOpenDynamicPromptsModal]);

  const onRandomPerImageClick = useCallback(() => {
    dispatch(modeChanged('random'));
    dispatch(randomRefreshModeChanged('per_image'));
  }, [dispatch]);

  const onRandomPerInvokeClick = useCallback(() => {
    dispatch(modeChanged('random'));
    dispatch(randomRefreshModeChanged('per_enqueue'));
  }, [dispatch]);

  const onRemoveWildcardOccurrence = useCallback(
    (occurrence: PromptWildcardOccurrence) => {
      const result = removePromptRange(prompt, occurrence.weight?.range ?? occurrence.range);
      applyPromptReplacement(result.prompt, { start: result.caret, end: result.caret });
    },
    [applyPromptReplacement, prompt]
  );

  const onRemoveWeightOccurrence = useCallback(
    (occurrence: PromptWeightOccurrence) => {
      const result = removePromptRange(prompt, occurrence.range);
      applyPromptReplacement(result.prompt, { start: result.caret, end: result.caret });
    },
    [applyPromptReplacement, prompt]
  );

  const onInspectorWildcardBehaviorAction = useCallback(
    (occurrence: PromptWildcardOccurrence, action: WildcardBehaviorAction) => {
      const intent = getWildcardBehaviorActionIntent(action, occurrence.path);

      if (intent.opensFixedValues) {
        if (!occurrence.wildcard) {
          return;
        }
        setFixedWildcardPath(occurrence.path);
        setFixedWildcardContext(null);
        setFixedWildcardOccurrence(occurrence);
        setActiveFixedValueIndex(0);
        loadWildcardValues({ path: occurrence.path, limit: 200 });
        return;
      }

      if (intent.removesPrompt) {
        onRemoveWildcardOccurrence(occurrence);
        return;
      }

      if (intent.replacement) {
        replaceOccurrenceRange(occurrence.range, intent.replacement);
      }
    },
    [loadWildcardValues, onRemoveWildcardOccurrence, replaceOccurrenceRange]
  );

  const hasPromptWorkbenchEntries = promptWorkbenchOccurrences.length > 0;
  const hasRandomWildcardOccurrences = promptWorkbenchOccurrences.some(
    (occurrence) => occurrence.type === 'wildcard' && occurrence.behavior === 'random'
  );
  const canShowWeightControls = capabilities.supportsAttentionWeights && hasPromptWorkbenchEntries;
  const canAdjustSelectionWeight = canShowWeightControls && selection.start !== selection.end;
  const hasPromptWorkbenchPanel = hasPromptWorkbenchEntries || diagnostics.length > 0;

  return (
    <Flex flexDir="column" gap={1} mt={1}>
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
                  {translate(wildcardStatusMessage)}
                </Text>
              </Flex>
            )}
            {wildcardOptions.map((wildcard, index) => (
              <Box key={wildcard.path}>
                <Flex
                  ref={wildcardOptionElementSetters[index]}
                  alignItems="center"
                  gap={1}
                  px={1}
                  minH={8}
                  bg={activeWildcardIndex === index ? 'base.700' : undefined}
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
                      <Text as="span" noOfLines={1} color="base.100" fontSize="sm" fontWeight="semibold">
                        {getWildcardDisplayPath(wildcard)}
                      </Text>
                    </Button>
                  </Tooltip>
                  <Text fontSize="sm" fontFamily="mono" color="base.400" textAlign="end" w={8} flexShrink={0}>
                    {wildcard.value_count}
                  </Text>
                  <PromptWildcardBehaviorMenu
                    ariaLabel={t('promptWorkbench.autocomplete.insertWithBehaviorAria', { path: wildcard.path })}
                    tooltip={t('promptWorkbench.actions.wildcardBehavior')}
                    iconType="random"
                    isActionable
                    canPickFixedValue
                    includeRemove={false}
                    onAction={onAutocompleteWildcardBehaviorAction(wildcard)}
                    onOpen={onAutocompleteBehaviorMenuOpen}
                    onClose={onAutocompleteBehaviorMenuClose}
                  />
                </Flex>
                {fixedWildcardPath === wildcard.path && (
                  <Flex flexDir="column" ps={4} pe={1} pb={1} gap={0.5} maxH={36} overflowY="auto">
                    {wildcardValuesResult.isFetching && (
                      <Text fontSize="xs" color="base.400">
                        {t('promptWorkbench.values.loading')}
                      </Text>
                    )}
                    {fixedWildcardValues?.map((value, index) => (
                      <Button
                        key={value}
                        ref={fixedValueElementSetters[index]}
                        size="xs"
                        variant="ghost"
                        justifyContent="flex-start"
                        h={7}
                        px={2}
                        bg={activeFixedValueIndex === index ? 'base.700' : undefined}
                        onMouseDown={onFixedValueMouseDown(value)}
                      >
                        <Text as="span" noOfLines={1} fontSize="sm">
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
            {completionContext.query
              ? t('promptWorkbench.autocomplete.matching', { query: completionContext.query })
              : t('promptWorkbench.autocomplete.localWildcards')}
          </Text>
        </Box>
      )}
      {hasPromptWorkbenchPanel && (
        <Box
          borderWidth={1}
          borderColor={PROMPT_INTENT_PANEL_BORDER}
          borderRadius="base"
          bg={PROMPT_INTENT_PANEL_BG}
          boxShadow="0 1px 0 rgba(255, 255, 255, 0.025) inset"
          overflow="hidden"
          p={1.5}
          data-testid="prompt-intent-panel"
        >
          <Flex flexDir="column" gap={1} mb={1.5}>
            <Flex alignItems="center" gap={1.5} minW={0}>
              <Box as={PiCubeBold} color="base.300" fontSize="0.95rem" flexShrink={0} />
              <Text color="base.100" fontSize="sm" fontWeight="semibold" lineHeight="short" noOfLines={1}>
                {t('promptWorkbench.panel.title')}
              </Text>
              {canShowWeightControls && (
                <Flex gap={1} ms="auto">
                  <Tooltip label={t(capabilities.attentionWeightsLabelKey)}>
                    <Button
                      size="xs"
                      variant="outline"
                      minW={6}
                      h={6}
                      isDisabled={!canAdjustSelectionWeight}
                      onMouseDown={onDecrementMouseDown}
                      onClick={onDecrementClick}
                    >
                      -
                    </Button>
                  </Tooltip>
                  <Tooltip label={t(capabilities.attentionWeightsLabelKey)}>
                    <Button
                      size="xs"
                      variant="outline"
                      minW={6}
                      h={6}
                      isDisabled={!canAdjustSelectionWeight}
                      onMouseDown={onIncrementMouseDown}
                      onClick={onIncrementClick}
                    >
                      +
                    </Button>
                  </Tooltip>
                </Flex>
              )}
            </Flex>
            <Flex gap={1} alignItems="center" flexWrap="wrap" minW={0}>
              {diagnostics.map((diagnostic) =>
                diagnostic.code === 'dynamic-active' && hasRandomWildcardOccurrences ? (
                  <Menu key={diagnostic.code}>
                    <Tooltip
                      label={t('promptWorkbench.diagnostics.changeRandomBehaviorTooltip', {
                        description: translate(diagnostic.description),
                      })}
                    >
                      <MenuButton as={Button} variant="unstyled" minW="unset" h="auto" cursor="pointer">
                        <PromptWorkbenchBadge tone={getDiagnosticBadgeTone(diagnostic.severity)}>
                          {translate(diagnostic.label)}
                        </PromptWorkbenchBadge>
                      </MenuButton>
                    </Tooltip>
                    <MenuList>
                      <MenuItem onClick={onRandomPerImageClick} title={t('promptWorkbench.header.randomImageTooltip')}>
                        {t('promptWorkbench.behavior.randomImageShort')}
                      </MenuItem>
                      <MenuItem
                        onClick={onRandomPerInvokeClick}
                        title={t('promptWorkbench.header.randomInvokeTooltip')}
                      >
                        {t('promptWorkbench.behavior.randomInvokeShort')}
                      </MenuItem>
                    </MenuList>
                  </Menu>
                ) : diagnostic.code === 'dynamic-active' ? (
                  <Tooltip
                    key={diagnostic.code}
                    label={t('promptWorkbench.diagnostics.openDynamicPreviewTooltip', {
                      description: translate(diagnostic.description),
                    })}
                  >
                    <Button
                      variant="unstyled"
                      minW="unset"
                      h="auto"
                      cursor="pointer"
                      onMouseDown={onOpenQueuedOutputsMouseDown}
                      onClick={onOpenQueuedOutputsClick}
                    >
                      <PromptWorkbenchBadge tone={getDiagnosticBadgeTone(diagnostic.severity)}>
                        {translate(diagnostic.label)}
                      </PromptWorkbenchBadge>
                    </Button>
                  </Tooltip>
                ) : (
                  <Tooltip key={diagnostic.code} label={translate(diagnostic.description)}>
                    <PromptWorkbenchBadge
                      tone={getDiagnosticBadgeTone(diagnostic.severity)}
                      icon={diagnostic.code === 'wildcards-found' ? <PiDiceFiveBold /> : undefined}
                    >
                      {translate(diagnostic.label)}
                    </PromptWorkbenchBadge>
                  </Tooltip>
                )
              )}
            </Flex>
          </Flex>
          {hasPromptWorkbenchEntries && (
            <Box>
              <PromptInspector
                occurrences={promptWorkbenchOccurrences}
                randomRefreshMode={dynamicPromptRandomRefreshMode}
                fixedWildcardOccurrenceId={activeFixedWildcardOccurrenceId}
                fixedWildcardValues={fixedWildcardValues}
                isFetchingFixedWildcardValues={wildcardValuesResult.isFetching}
                activeFixedValueIndex={activeFixedValueIndex}
                onSelectRange={focusPromptRange}
                onWildcardBehaviorAction={onInspectorWildcardBehaviorAction}
                onRemoveWeightOccurrence={onRemoveWeightOccurrence}
                onFixedValue={onInspectorFixedValue}
                setFixedValueElement={setFixedValueElement}
              />
            </Box>
          )}
        </Box>
      )}
    </Flex>
  );
});

PromptWorkbench.displayName = 'PromptWorkbench';

const getDiagnosticBadgeTone = (severity: PromptDiagnosticSeverity): PromptWorkbenchBadgeTone => {
  switch (severity) {
    case 'ok':
      return 'neutral';
    case 'warning':
      return 'warning';
    case 'error':
      return 'error';
    case 'info':
      return 'neutral';
  }
};
