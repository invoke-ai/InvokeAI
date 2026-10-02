import type { PromptTemplateSnapshot } from '@features/generation/core/promptTemplates';
import type { PromptTemplateRecord } from '@features/generation/data/promptTemplates';
import type { PendingPromptTemplateDraft } from '@features/generation/ui/promptTemplateDraftStore';

import { Popover, Portal, Text } from '@chakra-ui/react';
import { toPromptTemplateSnapshot } from '@features/generation/data/promptTemplates';
import { PromptTemplateEditor } from '@features/generation/ui/promptFields/PromptTemplateEditor';
import { PromptTemplatesPanel } from '@features/generation/ui/promptFields/PromptTemplatesPanel';
import { useOnPendingPromptTemplateDraft } from '@features/generation/ui/promptTemplateDraftStore';
import { isPromptTemplateMissing, usePromptTemplates } from '@features/generation/ui/usePromptTemplates';
import { IconButton } from '@platform/ui/Button';
import { MiddleTruncate } from '@platform/ui/MiddleTruncate';
import { PopoverContent } from '@platform/ui/Popover';
import { Tooltip } from '@platform/ui/Tooltip';
import { LayoutTemplateIcon } from 'lucide-react';
import { useCallback, useId, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

const POPOVER_POSITIONING_BOTTOM_END = { placement: 'bottom-end' } as const;

interface PromptTemplatesButtonProps {
  activeTemplate: PromptTemplateSnapshot | null;
  showSyntaxHighlighting: boolean;
  onApply: (template: PromptTemplateSnapshot | null) => void;
}

/** A null record creates a template; gallery input may prefill it. */
type EditorTarget = { record: PromptTemplateRecord | null; prefill?: PendingPromptTemplateDraft };

export const PromptTemplatesButton = ({
  activeTemplate,
  onApply,
  showSyntaxHighlighting,
}: PromptTemplatesButtonProps) => {
  const { t } = useTranslation();
  const triggerId = useId();
  const [isOpen, setIsOpen] = useState(false);
  const [editorTarget, setEditorTarget] = useState<EditorTarget | null>(null);
  // Enable the catalog query only for an open panel or applied template; any enabled observer triggers fetching.
  const catalog = usePromptTemplates({ isEnabled: isOpen || activeTemplate !== null });
  const popoverIds = useMemo(() => ({ trigger: triggerId }), [triggerId]);
  // A draft handed over from the gallery opens the editor straight away.
  useOnPendingPromptTemplateDraft(
    useCallback((prefill: PendingPromptTemplateDraft) => {
      setEditorTarget({ prefill, record: null });
      setIsOpen(true);
    }, [])
  );

  const handleOpenChange = useCallback((event: { open: boolean }) => {
    setIsOpen(event.open);

    // Reopen at the list rather than resume abandoned drafts.
    if (!event.open) {
      setEditorTarget(null);
    }
  }, []);

  const applyAndClose = useCallback(
    (template: PromptTemplateSnapshot | null) => {
      onApply(template);
      setIsOpen(false);
    },
    [onApply]
  );

  /** Deleting the applied template stops it applying, but the panel stays open. */
  const detachTemplate = useCallback(() => onApply(null), [onApply]);

  const startCreate = useCallback(() => setEditorTarget({ record: null }), []);
  const startEdit = useCallback((record: PromptTemplateRecord) => setEditorTarget({ record }), []);
  const closeEditor = useCallback(() => setEditorTarget(null), []);

  /** Refresh the applied snapshot after edits so displayed and generated text agree. */
  const handleSaved = useCallback(
    (saved: PromptTemplateRecord) => {
      if (activeTemplate?.id === saved.id) {
        onApply(toPromptTemplateSnapshot(saved));
      }

      setEditorTarget(null);
    },
    [activeTemplate?.id, onApply]
  );

  // Deleted templates still apply their stored snapshot with an explicit notification.
  const isMissing = isPromptTemplateMissing(catalog, activeTemplate);

  const tooltip = activeTemplate
    ? t(`widgets.generate.promptTemplates.${isMissing ? 'appliedMissing' : 'applied'}`, { name: activeTemplate.name })
    : t('widgets.generate.promptTemplates.title');

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
            aria-label={t('widgets.generate.promptTemplates.title')}
            // Quiet states only: dimmed while nothing is applied. No accent, no motion.
            opacity={activeTemplate ? undefined : 0.5}
            px="1"
            size="2xs"
            variant="ghost"
            w="auto"
          >
            <LayoutTemplateIcon />
            {activeTemplate ? (
              <MiddleTruncate
                as="span"
                fontSize="2xs"
                maxW="6rem"
                opacity={isMissing ? 0.5 : undefined}
                text={activeTemplate.name}
              />
            ) : (
              <Text as="span" fontSize="2xs">
                {t('widgets.generate.templatesButton')}
              </Text>
            )}
          </IconButton>
        </Popover.Trigger>
      </Tooltip>
      <Portal>
        <Popover.Positioner>
          <PopoverContent w="26rem">
            <Popover.Body p="2.5">
              {editorTarget ? (
                <PromptTemplateEditor
                  // Remount on handover target changes because editor drafts seed only on mount.
                  key={editorTarget.record?.id ?? 'new'}
                  catalog={catalog}
                  prefill={editorTarget.prefill}
                  showSyntaxHighlighting={showSyntaxHighlighting}
                  template={editorTarget.record}
                  onCancel={closeEditor}
                  onSaved={handleSaved}
                />
              ) : (
                <PromptTemplatesPanel
                  activeTemplate={activeTemplate}
                  isActiveTemplateMissing={isMissing}
                  catalog={catalog}
                  onApply={applyAndClose}
                  onDetach={detachTemplate}
                  onCreate={startCreate}
                  onEdit={startEdit}
                />
              )}
            </Popover.Body>
          </PopoverContent>
        </Popover.Positioner>
      </Portal>
    </Popover.Root>
  );
};
