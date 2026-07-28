import type { PromptTemplateSnapshot } from '@features/generation/core/promptTemplates';
import type { PromptTemplateRecord } from '@features/generation/data/promptTemplates';
import type { PendingPromptTemplateDraft } from '@features/generation/ui/promptTemplateDraftStore';

import { Popover, Portal, Text } from '@chakra-ui/react';
import { toPromptTemplateSnapshot } from '@features/generation/data/promptTemplates';
import { PromptTemplateEditor } from '@features/generation/ui/promptFields/PromptTemplateEditor';
import { PromptTemplatesPanel } from '@features/generation/ui/promptFields/PromptTemplatesPanel';
import { useOnPendingPromptTemplateDraft } from '@features/generation/ui/promptTemplateDraftStore';
import { usePromptTemplates } from '@features/generation/ui/usePromptTemplates';
import { IconButton } from '@platform/ui/Button';
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

/**
 * `record: null` means the editor is composing a new template; `prefill` carries
 * the prompts handed over from the gallery's "save as prompt template".
 */
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
  const catalog = usePromptTemplates();
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

    // Reopening should land on the list rather than resuming a half-written
    // template the user walked away from.
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

  /**
   * Editing the template that is currently applied has to refresh the snapshot,
   * or the panel would show the new text while the old one keeps generating.
   */
  const handleSaved = useCallback(
    (saved: PromptTemplateRecord) => {
      if (activeTemplate?.id === saved.id) {
        onApply(toPromptTemplateSnapshot(saved));
      }

      setEditorTarget(null);
    },
    [activeTemplate?.id, onApply]
  );

  const tooltip = activeTemplate
    ? t('widgets.generate.promptTemplates.applied', { name: activeTemplate.name })
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
            // Quiet states only: a dimmed icon when nothing is applied, the
            // template's name beside it when one is. No accent, no motion.
            opacity={activeTemplate ? undefined : 0.5}
            px={activeTemplate ? '1' : undefined}
            size="2xs"
            variant="ghost"
            w={activeTemplate ? 'auto' : undefined}
          >
            <LayoutTemplateIcon />
            {activeTemplate ? (
              <Text as="span" fontSize="2xs" maxW="6rem" truncate>
                {activeTemplate.name}
              </Text>
            ) : null}
          </IconButton>
        </Popover.Trigger>
      </Tooltip>
      <Portal>
        <Popover.Positioner>
          <Popover.Content bg="bg.muted" borderColor="border.emphasized" borderWidth="1px" w="26rem">
            <Popover.Body p="2.5">
              {editorTarget ? (
                <PromptTemplateEditor
                  // The editor seeds its draft on mount. Every other way in goes
                  // through the panel first, which unmounts it — but the gallery
                  // handover swaps the target in place, so without this a draft
                  // opened over an existing template kept that template's text
                  // and saved a duplicate of it.
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
                  catalog={catalog}
                  onApply={applyAndClose}
                  onDetach={detachTemplate}
                  onCreate={startCreate}
                  onEdit={startEdit}
                />
              )}
            </Popover.Body>
          </Popover.Content>
        </Popover.Positioner>
      </Portal>
    </Popover.Root>
  );
};
