import type { PromptTemplateSnapshot } from '@features/generation/core/promptTemplates';
import type { PromptTemplateRecord } from '@features/generation/data/promptTemplates';
import type { PromptTemplateCatalog } from '@features/generation/ui/usePromptTemplates';
import type { ChangeEvent } from 'react';

import { Box, HStack, Image, Input, Separator, Stack, Text } from '@chakra-ui/react';
import { searchCatalog } from '@features/generation/core/catalogSearch';
import { PROMPT_TEMPLATE_PLACEHOLDER } from '@features/generation/core/promptTemplates';
import { toPromptTemplateSnapshot } from '@features/generation/data/promptTemplates';
import { useGenerationUi } from '@features/generation/ui/GenerationUiContext';
import { PANEL_HEADER_CONTROL_HEIGHT, PromptPanelHeader } from '@features/generation/ui/promptFields/PromptPanelHeader';
import { downloadBlob } from '@platform/browser/downloadBlob';
import { getApiErrorMessage } from '@platform/transport/http';
import { Button, IconButton } from '@platform/ui/Button';
import { ConfirmDialog } from '@platform/ui/ConfirmDialog';
import { Scrollable } from '@platform/ui/Scrollable';
import { Tooltip } from '@platform/ui/Tooltip';
import { DownloadIcon, ImageIcon, PencilIcon, PlusIcon, TrashIcon, UploadIcon } from 'lucide-react';
import { useCallback, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

const THUMBNAIL_SIZE = '7';

interface PromptTemplatesPanelProps {
  catalog: PromptTemplateCatalog;
  activeTemplate: PromptTemplateSnapshot | null;
  onApply: (template: PromptTemplateSnapshot | null) => void;
  /**
   * Stop applying the active template without dismissing the panel — for when it
   * has just been deleted, which is not the same intent as choosing to clear it.
   */
  onDetach: () => void;
  onEdit: (template: PromptTemplateRecord) => void;
  onCreate: () => void;
}

/** A template's prose is its two prompts; `searchCatalog` handles the name. */
const getTemplateProse = (template: PromptTemplateRecord): readonly string[] => [
  template.positivePrompt,
  template.negativePrompt,
];

export const PromptTemplatesPanel = ({
  activeTemplate,
  catalog,
  onApply,
  onDetach,
  onCreate,
  onEdit,
}: PromptTemplatesPanelProps) => {
  const { t } = useTranslation();
  const { capabilities, notifications } = useGenerationUi();
  const canManagePromptTemplates = capabilities.canManagePromptTemplates;
  const [searchTerm, setSearchTerm] = useState('');
  const [pendingDelete, setPendingDelete] = useState<PromptTemplateRecord | null>(null);

  const userTemplates = useMemo(
    () => searchCatalog(catalog.userTemplates, searchTerm, getTemplateProse),
    [catalog.userTemplates, searchTerm]
  );
  const defaultTemplates = useMemo(
    () => searchCatalog(catalog.defaultTemplates, searchTerm, getTemplateProse),
    [catalog.defaultTemplates, searchTerm]
  );
  const hasResults = userTemplates.length > 0 || defaultTemplates.length > 0;

  const handleSearchChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => setSearchTerm(event.currentTarget.value),
    []
  );

  const clearActiveTemplate = useCallback(() => onApply(null), [onApply]);
  const closeDeleteDialog = useCallback(() => setPendingDelete(null), []);

  /**
   * The dialog closes either way, so a failure has nowhere inline to land — it
   * goes to the notification centre like the other Generation errors.
   */
  const confirmDelete = useCallback(async () => {
    if (!pendingDelete) {
      return;
    }

    try {
      await catalog.remove(pendingDelete.id);

      // Deleting the template that is applied would otherwise leave it silently
      // shaping every prompt with no way to reach it. Detached rather than
      // cleared: the user is in the middle of managing the list, and closing the
      // panel out from under them is not what they asked for.
      if (activeTemplate?.id === pendingDelete.id) {
        onDetach();
      }
    } catch (caught) {
      notifications.reportError({
        area: 'delete-prompt-template',
        message: getApiErrorMessage(caught, t('widgets.generate.promptTemplates.couldNotDelete')),
        namespace: 'generation',
      });
    }
  }, [activeTemplate, catalog, notifications, onDetach, pendingDelete, t]);

  return (
    <Stack gap="2">
      <PromptPanelHeader label={t('widgets.generate.promptTemplates.title')}>
        <Button h={PANEL_HEADER_CONTROL_HEIGHT} size="2xs" variant="ghost" onClick={onCreate}>
          <PlusIcon />
          {t('widgets.generate.promptTemplates.newTemplate')}
        </Button>
      </PromptPanelHeader>

      <Input
        aria-label={t('widgets.generate.promptTemplates.search')}
        placeholder={t('widgets.generate.promptTemplates.search')}
        size="xs"
        value={searchTerm}
        onChange={handleSearchChange}
      />
      <Separator />

      <Scrollable h="16rem" label={t('widgets.generate.promptTemplates.title')}>
        {catalog.isLoading ? (
          <PanelMessage>{t('widgets.generate.promptTemplates.loading')}</PanelMessage>
        ) : !hasResults ? (
          <PanelMessage>
            {searchTerm.trim()
              ? t('widgets.generate.promptTemplates.noMatches')
              : t('widgets.generate.promptTemplates.noTemplatesYet')}
          </PanelMessage>
        ) : (
          <Stack gap="2">
            <TemplateGroup
              activeTemplateId={activeTemplate?.id ?? null}
              label={t('widgets.generate.promptTemplates.yourTemplates')}
              templates={userTemplates}
              onApply={onApply}
              onDelete={setPendingDelete}
              onEdit={onEdit}
            />
            <TemplateGroup
              activeTemplateId={activeTemplate?.id ?? null}
              label={t('widgets.generate.promptTemplates.defaultTemplates')}
              templates={defaultTemplates}
              onApply={onApply}
            />
          </Stack>
        )}
      </Scrollable>

      {activeTemplate || canManagePromptTemplates ? (
        <>
          <Separator />
          <HStack gap="1" justify="space-between">
            {canManagePromptTemplates ? <PromptTemplateTransferActions catalog={catalog} /> : <Box />}
            {activeTemplate ? (
              <Button size="2xs" variant="ghost" onClick={clearActiveTemplate}>
                {t('widgets.generate.promptTemplates.clearApplied', { name: activeTemplate.name })}
              </Button>
            ) : null}
          </HStack>
        </>
      ) : null}

      <ConfirmDialog
        body={t('widgets.generate.promptTemplates.deleteBody', { name: pendingDelete?.name ?? '' })}
        confirmLabel={t('common.delete')}
        isOpen={pendingDelete !== null}
        title={t('widgets.generate.promptTemplates.deleteTitle')}
        onClose={closeDeleteDialog}
        onConfirm={confirmDelete}
      />
    </Stack>
  );
};

/**
 * Bulk transfer, admin-only on the backend. Export streams CSV and import accepts
 * the CSV/JSON shapes the backend's parser documents. Both go through the
 * transport rather than an anchor download, which would carry no auth header.
 */
const PromptTemplateTransferActions = ({ catalog }: { catalog: PromptTemplateCatalog }) => {
  const { t } = useTranslation();
  const { notifications } = useGenerationUi();
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [isBusy, setIsBusy] = useState(false);

  const reportError = useCallback(
    (area: string, caught: unknown, fallback: string) =>
      notifications.reportError({
        area,
        message: getApiErrorMessage(caught, fallback),
        namespace: 'generation',
      }),
    [notifications]
  );

  const runImport = useCallback(
    async (file: File) => {
      setIsBusy(true);

      try {
        await catalog.importFile(file);
      } catch (caught) {
        reportError('import-prompt-templates', caught, t('widgets.generate.promptTemplates.couldNotImport'));
      } finally {
        setIsBusy(false);
      }
    },
    [catalog, reportError, t]
  );

  const runExport = useCallback(async () => {
    setIsBusy(true);

    try {
      downloadBlob(await catalog.exportCsv(), 'prompt_templates.csv');
    } catch (caught) {
      reportError('export-prompt-templates', caught, t('widgets.generate.promptTemplates.couldNotExport'));
    } finally {
      setIsBusy(false);
    }
  }, [catalog, reportError, t]);

  const pickFile = useCallback(() => fileInputRef.current?.click(), []);
  const handleExport = useCallback(() => void runExport(), [runExport]);

  const handleFileChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => {
      const file = event.currentTarget.files?.[0];

      if (file) {
        void runImport(file);
      }

      event.currentTarget.value = '';
    },
    [runImport]
  );

  return (
    <HStack gap="0.5">
      <Tooltip content={t('widgets.generate.promptTemplates.importHelp')}>
        <Button disabled={isBusy} size="2xs" variant="ghost" onClick={pickFile}>
          <UploadIcon />
          {t('widgets.generate.promptTemplates.import')}
        </Button>
      </Tooltip>
      <Tooltip content={t('widgets.generate.promptTemplates.exportHelp')}>
        <Button disabled={isBusy} size="2xs" variant="ghost" onClick={handleExport}>
          <DownloadIcon />
          {t('widgets.generate.promptTemplates.export')}
        </Button>
      </Tooltip>
      <input
        accept=".csv,.json,text/csv,application/json"
        hidden
        ref={fileInputRef}
        type="file"
        onChange={handleFileChange}
      />
    </HStack>
  );
};

const PanelMessage = ({ children }: { children: string }) => (
  <HStack h="full" justify="center" minH="8rem">
    <Text color="fg.subtle" fontSize="xs">
      {children}
    </Text>
  </HStack>
);

const TemplateGroup = ({
  activeTemplateId,
  label,
  onApply,
  onDelete,
  onEdit,
  templates,
}: {
  activeTemplateId: string | null;
  label: string;
  templates: PromptTemplateRecord[];
  onApply: (template: PromptTemplateSnapshot) => void;
  onDelete?: (template: PromptTemplateRecord) => void;
  onEdit?: (template: PromptTemplateRecord) => void;
}) => {
  if (templates.length === 0) {
    return null;
  }

  return (
    <Stack gap="0">
      <Text color="fg.subtle" fontSize="2xs" fontWeight="700" px="2" textTransform="uppercase">
        {label}
      </Text>
      {templates.map((template) => (
        <TemplateRow
          key={template.id}
          isActive={template.id === activeTemplateId}
          template={template}
          onApply={onApply}
          onDelete={onDelete}
          onEdit={onEdit}
        />
      ))}
    </Stack>
  );
};

const TemplateRow = ({
  isActive,
  onApply,
  onDelete,
  onEdit,
  template,
}: {
  isActive: boolean;
  template: PromptTemplateRecord;
  onApply: (template: PromptTemplateSnapshot) => void;
  onDelete?: (template: PromptTemplateRecord) => void;
  onEdit?: (template: PromptTemplateRecord) => void;
}) => {
  const { t } = useTranslation();
  const handleApply = useCallback(() => onApply(toPromptTemplateSnapshot(template)), [onApply, template]);
  const handleEdit = useCallback(() => onEdit?.(template), [onEdit, template]);
  const handleDelete = useCallback(() => onDelete?.(template), [onDelete, template]);
  // The prompt with its placeholder is the whole content of a template, so the
  // row shows it rather than a name alone.
  const summary = template.positivePrompt || PROMPT_TEMPLATE_PLACEHOLDER;

  return (
    <HStack align="center" gap="1" pr="1">
      <Button
        alignItems="center"
        flex="1"
        h="auto"
        justifyContent="start"
        minW="0"
        px="2"
        py="1.5"
        size="xs"
        variant="ghost"
        onClick={handleApply}
      >
        <TemplateThumbnail template={template} />
        <Stack align="start" flex="1" gap="0" minW="0">
          <Text as="span" color={isActive ? 'fg' : 'fg.muted'} fontSize="xs" fontWeight={isActive ? '600' : '400'}>
            {template.name}
          </Text>
          <Text as="span" color="fg.subtle" fontFamily="mono" fontSize="2xs" truncate>
            {summary}
          </Text>
        </Stack>
      </Button>
      {onEdit ? (
        <Tooltip content={t('common.edit')}>
          <IconButton aria-label={t('common.edit')} size="2xs" variant="ghost" onClick={handleEdit}>
            <PencilIcon />
          </IconButton>
        </Tooltip>
      ) : null}
      {onDelete ? (
        <Tooltip content={t('common.delete')}>
          <IconButton
            aria-label={t('common.delete')}
            colorPalette="red"
            size="2xs"
            variant="ghost"
            onClick={handleDelete}
          >
            <TrashIcon />
          </IconButton>
        </Tooltip>
      ) : null}
    </HStack>
  );
};

const TemplateThumbnail = ({ template }: { template: PromptTemplateRecord }) =>
  template.imageUrl ? (
    <Image alt="" boxSize={THUMBNAIL_SIZE} flexShrink="0" objectFit="cover" rounded="sm" src={template.imageUrl} />
  ) : (
    <Box
      alignItems="center"
      bg="bg.emphasized"
      boxSize={THUMBNAIL_SIZE}
      color="fg.subtle"
      display="flex"
      flexShrink="0"
      justifyContent="center"
      rounded="sm"
    >
      <ImageIcon size={12} />
    </Box>
  );
