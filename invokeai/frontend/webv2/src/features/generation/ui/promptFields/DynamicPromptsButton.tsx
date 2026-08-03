import type { DynamicPromptsFieldConfig } from '@features/generation/ui/promptFields/DynamicPromptsPanel';

import { Popover, Portal, SegmentGroup, Stack, Text } from '@chakra-ui/react';
import { DynamicPromptsPanel } from '@features/generation/ui/promptFields/DynamicPromptsPanel';
import { WildcardsPanel } from '@features/generation/ui/promptFields/WildcardsPanel';
import { useDynamicPrompts } from '@features/generation/ui/useDynamicPrompts';
import { useWildcards } from '@features/generation/ui/useWildcards';
import { IconButton } from '@platform/ui/Button';
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
  const [isOpen, setIsOpen] = useState(false);
  const [tab, setTab] = useState<'preview' | 'wildcards'>('preview');
  const expansion = useDynamicPrompts(positivePrompt, config);
  const catalog = useWildcards();
  const popoverIds = useMemo(() => ({ trigger: triggerId }), [triggerId]);

  const handleOpenChange = useCallback((event: { open: boolean }) => setIsOpen(event.open), []);
  const handleTabChange = useCallback(
    (event: { value: string | null }) => setTab(event.value === 'wildcards' ? 'wildcards' : 'preview'),
    []
  );
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
      { label: t('widgets.generate.dynamicPrompts.preview'), value: 'preview' },
      { label: t('widgets.generate.dynamicPrompts.wildcards'), value: 'wildcards' },
    ],
    [t]
  );

  const tooltip = expansion.isDynamic
    ? t('widgets.generate.dynamicPrompts.showPrompts')
    : t('widgets.generate.dynamicPrompts.noDynamicSyntax');
  // Quiet states only: an em-dash while the expansion is in flight, an error tint
  // when it failed. No spinner, no animation on a control this small.
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
            // `undefined` is not "leave it alone": Chakra's IconButton spreads
            // incoming props over its own `px: '0'`, so an explicit undefined
            // clobbers it and the size recipe's `px: '2'` padded the icon-only
            // button out to 32x24 beside its 24x24 neighbours.
            px={countLabel ? '1' : '0'}
            size="2xs"
            variant="ghost"
            w={countLabel ? 'auto' : undefined}
          >
            <BracesIcon />
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
          <Popover.Content bg="bg.muted" borderColor="border.emphasized" borderWidth="1px" w="26rem">
            <Popover.Body p="2.5">
              <Stack gap="2.5">
                {/* `alignSelf` keeps the tabs to their content width; stretched across
                    the popover they read as a header band rather than a control. */}
                <SegmentGroup.Root alignSelf="start" size="xs" value={tab} onValueChange={handleTabChange}>
                  <SegmentGroup.Indicator />
                  <SegmentGroup.Items items={tabItems} />
                </SegmentGroup.Root>
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
            </Popover.Body>
          </Popover.Content>
        </Popover.Positioner>
      </Portal>
    </Popover.Root>
  );
};
