import type { DynamicPromptsFieldConfig } from '@features/generation/ui/promptFields/DynamicPromptsPanel';
import type { PromptEditor } from '@features/generation/ui/promptFields/promptEditor';
import type { DynamicPromptsExpansion } from '@features/generation/ui/useDynamicPrompts';

import { Popover, Portal, Stack, Text } from '@chakra-ui/react';
import { DynamicPromptsPanel } from '@features/generation/ui/promptFields/DynamicPromptsPanel';
import { useFocusAfterPromptWrite } from '@features/generation/ui/promptFields/promptEditor';
import { useWildcards } from '@features/generation/ui/useWildcards';
import { IconButton } from '@platform/ui/Button';
import { PopoverContent } from '@platform/ui/Popover';
import { SegmentTabs, segmentTabsPanelId, segmentTabsTabId } from '@platform/ui/SegmentTabs';
import { Tooltip } from '@platform/ui/Tooltip';
import { BracesIcon } from 'lucide-react';
import { lazy, Suspense, useCallback, useId, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

const POPOVER_POSITIONING_BOTTOM_END = { placement: 'bottom-end' } as const;

// The wildcard manager and its list load with the popover, not with every prompt field at editor boot.
const loadWildcardsPanel = () => import('@features/generation/ui/promptFields/WildcardsPanel');
const WildcardsPanel = lazy(() => loadWildcardsPanel().then((module) => ({ default: module.WildcardsPanel })));
const TABULAR_NUMS = { fontVariantNumeric: 'tabular-nums' } as const;

interface DynamicPromptsButtonProps {
  config: DynamicPromptsFieldConfig;
  batchCount: number;
  /** The expansion of the prompt as it generates (inside its template). */
  expansion: DynamicPromptsExpansion;
  /** Where focus returns after a pick lands in the prompt. */
  promptEditor: Pick<PromptEditor, 'element'>;
  showSyntaxHighlighting: boolean;
  /** Expanded prompts keep only attention syntax, so they follow the model rather than the authoring preference. */
  highlightExpandedPrompts: boolean;
  onUsePrompt: (prompt: string) => void;
  onInsertText: (text: string) => void;
}

export const DynamicPromptsButton = ({
  batchCount,
  config,
  expansion,
  highlightExpandedPrompts,
  onInsertText,
  onUsePrompt,
  promptEditor,
  showSyntaxHighlighting,
}: DynamicPromptsButtonProps) => {
  const { t } = useTranslation();
  const triggerId = useId();
  const tabsIdBase = useId();
  const [isOpen, setIsOpen] = useState(false);
  const [tab, setTab] = useState<'preview' | 'wildcards'>('preview');
  const catalog = useWildcards();
  const popoverIds = useMemo(() => ({ trigger: triggerId }), [triggerId]);
  const { finalFocusEl, keepFocusAfterWrite } = useFocusAfterPromptWrite(promptEditor);

  const handleOpenChange = useCallback((event: { open: boolean }) => {
    if (event.open) {
      // Ready before the Wildcards tab is chosen, so it opens without a loading gap.
      void loadWildcardsPanel();
    }
    setIsOpen(event.open);
  }, []);
  const handleTabChange = useCallback((value: 'preview' | 'wildcards') => setTab(value), []);
  const closeWith = useCallback(
    (apply: (value: string) => void) => (value: string) => {
      apply(value);
      keepFocusAfterWrite();
      setIsOpen(false);
    },
    [keepFocusAfterWrite]
  );
  const handleUsePrompt = useMemo(() => closeWith(onUsePrompt), [closeWith, onUsePrompt]);
  const handleInsert = useMemo(() => closeWith(onInsertText), [closeWith, onInsertText]);

  const tabItems = useMemo(
    () => [
      { id: 'preview' as const, label: t('widgets.generate.dynamicPrompts.preview') },
      { id: 'wildcards' as const, label: t('widgets.generate.dynamicPrompts.wildcards') },
    ],
    [t]
  );

  const tooltip = !config.enabled
    ? t('widgets.generate.dynamicPrompts.offTooltip')
    : expansion.isDynamic
      ? t('widgets.generate.dynamicPrompts.showPrompts')
      : t('widgets.generate.dynamicPrompts.noDynamicSyntax');
  const countLabel = !config.enabled
    ? t('widgets.generate.dynamicPrompts.off')
    : !expansion.isDynamic
      ? null
      : expansion.isLoading
        ? '—'
        : String(expansion.count);

  return (
    <Popover.Root
      finalFocusEl={finalFocusEl}
      ids={popoverIds}
      lazyMount
      open={isOpen}
      positioning={POPOVER_POSITIONING_BOTTOM_END}
      unmountOnExit
      onOpenChange={handleOpenChange}
    >
      <Tooltip content={tooltip} ids={popoverIds}>
        <Popover.Trigger asChild>
          <IconButton
            aria-label={
              config.enabled
                ? t('widgets.generate.dynamicPrompts.showPrompts')
                : t('widgets.generate.dynamicPrompts.offLabel')
            }
            color={expansion.isError ? 'fg.error' : undefined}
            opacity={expansion.isDynamic ? undefined : 0.5}
            px="1"
            size="sm"
            variant="ghost"
            w="auto"
          >
            <BracesIcon />
            <Text as="span" fontSize="xs">
              {t('widgets.generate.dynamicButton')}
            </Text>
            {countLabel ? (
              <Text as="span" css={TABULAR_NUMS} fontSize="xs">
                {countLabel}
              </Text>
            ) : null}
          </IconButton>
        </Popover.Trigger>
      </Tooltip>
      <Portal>
        <Popover.Positioner>
          <PopoverContent w="26rem">
            <Popover.Body p="2.5">
              <Stack gap="1.5">
                <SegmentTabs
                  activeId={tab}
                  ariaLabel={t('widgets.generate.dynamicPrompts.title')}
                  idBase={tabsIdBase}
                  isCompact
                  tabs={tabItems}
                  onSelect={handleTabChange}
                />
                <Stack
                  aria-labelledby={segmentTabsTabId(tabsIdBase, tab)}
                  gap="2.5"
                  id={segmentTabsPanelId(tabsIdBase)}
                  role="tabpanel"
                >
                  {tab === 'preview' ? (
                    <DynamicPromptsPanel
                      batchCount={batchCount}
                      config={config}
                      expansion={expansion}
                      showSyntaxHighlighting={highlightExpandedPrompts}
                      onUsePrompt={handleUsePrompt}
                    />
                  ) : (
                    <Suspense fallback={null}>
                      <WildcardsPanel
                        catalog={catalog}
                        showSyntaxHighlighting={showSyntaxHighlighting}
                        onInsert={handleInsert}
                      />
                    </Suspense>
                  )}
                </Stack>
              </Stack>
            </Popover.Body>
          </PopoverContent>
        </Popover.Positioner>
      </Portal>
    </Popover.Root>
  );
};
