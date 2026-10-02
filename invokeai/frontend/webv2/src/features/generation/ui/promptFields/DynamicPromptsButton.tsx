import type { DynamicPromptsFieldConfig } from '@features/generation/ui/promptFields/DynamicPromptsPanel';

import { Popover, Portal, Stack, Text } from '@chakra-ui/react';
import { DynamicPromptsPanel } from '@features/generation/ui/promptFields/DynamicPromptsPanel';
import { WildcardsPanel } from '@features/generation/ui/promptFields/WildcardsPanel';
import { useDynamicPrompts } from '@features/generation/ui/useDynamicPrompts';
import { useWildcards } from '@features/generation/ui/useWildcards';
import { IconButton } from '@platform/ui/Button';
import { PopoverContent } from '@platform/ui/Popover';
import { SegmentTabs, segmentTabsPanelId, segmentTabsTabId } from '@platform/ui/SegmentTabs';
import { Tooltip } from '@platform/ui/Tooltip';
import { BracesIcon } from 'lucide-react';
import { useCallback, useId, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

const POPOVER_POSITIONING_BOTTOM_END = { placement: 'bottom-end' } as const;
const TABULAR_NUMS = { fontVariantNumeric: 'tabular-nums' } as const;

interface DynamicPromptsButtonProps {
  config: DynamicPromptsFieldConfig;
  batchCount: number;
  positivePrompt: string;
  showSyntaxHighlighting: boolean;
  onUsePrompt: (prompt: string) => void;
  onInsertText: (text: string) => void;
}

export const DynamicPromptsButton = ({
  batchCount,
  config,
  onInsertText,
  onUsePrompt,
  positivePrompt,
  showSyntaxHighlighting,
}: DynamicPromptsButtonProps) => {
  const { t } = useTranslation();
  const triggerId = useId();
  const tabsIdBase = useId();
  const [isOpen, setIsOpen] = useState(false);
  const [tab, setTab] = useState<'preview' | 'wildcards'>('preview');
  const expansion = useDynamicPrompts(positivePrompt, config);
  const catalog = useWildcards();
  const popoverIds = useMemo(() => ({ trigger: triggerId }), [triggerId]);

  const handleOpenChange = useCallback((event: { open: boolean }) => setIsOpen(event.open), []);
  const handleTabChange = useCallback((value: 'preview' | 'wildcards') => setTab(value), []);
  const closeWith = useCallback(
    (apply: (value: string) => void) => (value: string) => {
      apply(value);
      setIsOpen(false);
    },
    []
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

  const tooltip = expansion.isDynamic
    ? t('widgets.generate.dynamicPrompts.showPrompts')
    : t('widgets.generate.dynamicPrompts.noDynamicSyntax');
  const countLabel = !expansion.isDynamic ? null : expansion.isLoading ? '—' : String(expansion.count);

  return (
    <Popover.Root
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
            aria-label={t('widgets.generate.dynamicPrompts.showPrompts')}
            color={expansion.isError ? 'fg.error' : undefined}
            opacity={expansion.isDynamic ? undefined : 0.5}
            px="1"
            size="2xs"
            variant="ghost"
            w="auto"
          >
            <BracesIcon />
            <Text as="span" fontSize="2xs">
              {t('widgets.generate.dynamicButton')}
            </Text>
            {countLabel ? (
              <Text as="span" css={TABULAR_NUMS} fontSize="2xs">
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
                      showSyntaxHighlighting={showSyntaxHighlighting}
                      onUsePrompt={handleUsePrompt}
                    />
                  ) : (
                    <WildcardsPanel
                      catalog={catalog}
                      showSyntaxHighlighting={showSyntaxHighlighting}
                      onInsert={handleInsert}
                    />
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
