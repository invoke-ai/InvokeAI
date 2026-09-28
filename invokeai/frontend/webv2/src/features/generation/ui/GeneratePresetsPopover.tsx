import type { GenerateSettings } from '@features/generation/core/types';
import type { ChangeEvent } from 'react';

import { HStack, Icon, Input, InputGroup, Popover, Portal, Separator, Stack, Text } from '@chakra-ui/react';
import {
  getGenerateModelSelectionResult,
  isSupportedGenerateModel,
} from '@features/generation/core/baseGenerationPolicies';
import { GENERATE_TOOL_MODEL_PICK_KEYS, normalizeGenerateSettings } from '@features/generation/core/settings';
import { resolveGenerateWidgetValues } from '@features/generation/settings';
import {
  Button,
  ConfirmDialog,
  IconButton,
  PopoverContent,
  RenameDialog,
  Row,
  Scrollable,
  Tooltip,
} from '@platform/ui';
import { BookmarkIcon, CheckIcon, PencilIcon, PlusIcon, SearchIcon, Trash2Icon } from 'lucide-react';
import { useCallback, useId, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

import type { GeneratePresetRecord } from './GenerationUiContext';

import { flushGenerateDrafts } from './generateDraftRegistry';
import { getGenerateFormCommitPatch } from './generateFormViewModel';
import { useGenerateValues, useGenerationUi } from './GenerationUiContext';
import { notifyGenerateModelSelectionCleared } from './modelSelectionNotice';
import { PANEL_HEADER_CONTROL_HEIGHT, PromptPanelHeader } from './promptFields/PromptPanelHeader';

const POPOVER_POSITIONING = { placement: 'bottom-end' } as const;
const SEARCH_START_ELEMENT = <Icon as={SearchIcon} boxSize="3" color="fg.subtle" />;
// The rows sit on the popover's `bg.muted` surface; hover one surface step up.
const PRESET_ROW_HOVER_PROPS = { bg: 'bg.emphasized/60' };
const PRESET_ROW_SEPARATOR = <Separator borderColor="border.subtle" />;
/** Below this many presets, a search box is more furniture than help. */
const SEARCH_VISIBLE_MIN_PRESETS = 6;

type PresetDialogState = { mode: 'save' } | { mode: 'rename'; preset: GeneratePresetRecord };

/** Ignore presentation and non-applied keys, including seed unless its mode is fixed. */
const getPresetComparisonKey = (settings: GenerateSettings): string => {
  const comparable: Record<string, unknown> = { ...settings };

  delete comparable.batchCount;

  for (const key of GENERATE_TOOL_MODEL_PICK_KEYS) {
    delete comparable[key];
  }

  delete comparable.negativePromptHeightPx;
  delete comparable.positivePromptHeightPx;
  delete comparable.promptTemplateViewMode;

  if (settings.seedMode !== 'fixed') {
    delete comparable.seed;
  }

  return JSON.stringify(comparable);
};

const PresetRow = ({
  isActive,
  preset,
  onApply,
  onDelete,
  onRename,
}: {
  /** The current settings still equal this preset's snapshot. */
  isActive: boolean;
  preset: GeneratePresetRecord;
  onApply: (preset: GeneratePresetRecord) => void;
  onDelete: (preset: GeneratePresetRecord) => void;
  onRename: (preset: GeneratePresetRecord) => void;
}) => {
  const { t } = useTranslation();
  const handleApply = useCallback(() => onApply(preset), [onApply, preset]);
  const handleRename = useCallback(() => onRename(preset), [onRename, preset]);
  const handleDelete = useCallback(() => onDelete(preset), [onDelete, preset]);

  return (
    <HStack gap="0.5">
      <Row
        aria-current={isActive || undefined}
        asChild
        flex="1"
        h="7"
        justifyContent="start"
        minW="0"
        px="2"
        rounded="control"
        _hover={PRESET_ROW_HOVER_PROPS}
      >
        <button type="button" onClick={handleApply}>
          <Icon as={BookmarkIcon} boxSize="3.5" color="fg.subtle" flexShrink={0} />
          <Text flex="1" fontSize="xs" fontWeight={isActive ? '600' : undefined} minW="0" textAlign="start" truncate>
            {preset.label}
          </Text>

          {isActive ? <Icon as={CheckIcon} boxSize="3.5" color="accent.fg" flexShrink={0} /> : null}
        </button>
      </Row>
      <Tooltip content={t('common.rename')}>
        <IconButton
          aria-label={t('widgets.generate.renamePresetNamed', { name: preset.label })}
          size="xs"
          variant="ghost"
          onClick={handleRename}
        >
          <PencilIcon />
        </IconButton>
      </Tooltip>
      <Tooltip content={t('common.delete')}>
        <IconButton
          aria-label={t('widgets.generate.deletePresetNamed', { name: preset.label })}
          colorPalette="red"
          size="xs"
          variant="ghost"
          onClick={handleDelete}
        >
          <Trash2Icon />
        </IconButton>
      </Tooltip>
    </HStack>
  );
};

/** Catalog changes use the same resolve/patch path as model selection. */
export const GeneratePresetsPopover = () => {
  const { i18n, t } = useTranslation();
  const ui = useGenerationUi();
  const [isOpen, setIsOpen] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [dialog, setDialog] = useState<PresetDialogState | null>(null);
  const [pendingDelete, setPendingDelete] = useState<GeneratePresetRecord | null>(null);
  // Tooltip and Popover must share the trigger ID to preserve anchoring.
  const triggerId = useId();
  const triggerIds = useMemo(() => ({ trigger: triggerId }), [triggerId]);

  const models = ui.models.catalog;
  const projectId = ui.project.activeProjectId;
  const presets = ui.presets.presets;
  const supportedModels = useMemo(() => models.filter(isSupportedGenerateModel), [models]);
  const generateValues = useGenerateValues();
  const settings = useMemo(() => normalizeGenerateSettings(generateValues), [generateValues]);
  const canSave = supportedModels.some((model) => model.key === settings?.modelKey);

  const filteredPresets = useMemo(() => {
    const term = searchTerm.trim().toLowerCase();

    return term ? presets.filter((preset) => preset.label.toLowerCase().includes(term)) : presets;
  }, [presets, searchTerm]);

  // Serialized equality requires canonical key order.
  const presetKeys = useMemo(
    () =>
      presets.map((preset) => {
        const normalized = normalizeGenerateSettings(preset.values);

        return normalized ? getPresetComparisonKey(normalized) : null;
      }),
    [presets]
  );
  const activePreset = useMemo(() => {
    if (!settings) {
      return null;
    }

    const currentKey = getPresetComparisonKey(settings);
    const index = presetKeys.findIndex((key) => key !== null && key === currentKey);

    return index >= 0 ? (presets[index] ?? null) : null;
  }, [presetKeys, presets, settings]);

  const handleOpenChange = useCallback((event: { open: boolean }) => {
    setIsOpen(event.open);

    if (!event.open) {
      setSearchTerm('');
    }
  }, []);
  const handleSearchChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => setSearchTerm(event.currentTarget.value),
    []
  );

  const applyPreset = useCallback(
    (record: GeneratePresetRecord) => {
      const normalized = normalizeGenerateSettings(record.values);

      if (!normalized) {
        ui.notifications.error(t('widgets.generate.presetUnreadable'), record.label);
        return;
      }

      const model = supportedModels.find((candidate) => candidate.key === normalized.modelKey);

      if (!model) {
        ui.notifications.error(t('widgets.generate.presetModelMissing'), record.label);
        return;
      }

      const result = getGenerateModelSelectionResult({ currentValues: normalized, model, models });
      const resolved = resolveGenerateWidgetValues({ models, storedValues: { ...result.settings, model } });

      if (!resolved) {
        ui.notifications.error(t('widgets.generate.presetUnreadable'), record.label);
        return;
      }

      ui.settings.patchGenerateSettings(getGenerateFormCommitPatch(resolved.values), projectId);
      notifyGenerateModelSelectionCleared({
        clearedLabels: result.clearedLabels,
        locale: i18n.resolvedLanguage,
        modelName: model.name,
        notifications: ui.notifications,
        t,
      });
      setIsOpen(false);
    },
    [i18n.resolvedLanguage, models, projectId, supportedModels, t, ui.notifications, ui.settings]
  );

  const openSaveDialog = useCallback(() => setDialog({ mode: 'save' }), []);
  const openRenameDialog = useCallback((preset: GeneratePresetRecord) => setDialog({ mode: 'rename', preset }), []);
  const closeDialog = useCallback(() => setDialog(null), []);
  const handleDialogSubmit = useCallback(
    (label: string) => {
      if (dialog?.mode === 'rename') {
        ui.presets.rename(dialog.preset.id, label);
        return;
      }

      // Flush so the snapshot carries what the user sees, not a debounce behind it.
      flushGenerateDrafts();
      const snapshot = normalizeGenerateSettings(ui.generateValues.getSnapshot());

      if (snapshot) {
        ui.presets.save(label, { ...snapshot });
      }
    },
    [dialog, ui.generateValues, ui.presets]
  );

  const cancelDelete = useCallback(() => setPendingDelete(null), []);
  const confirmDelete = useCallback(() => {
    if (pendingDelete) {
      ui.presets.remove(pendingDelete.id);
    }

    setPendingDelete(null);
  }, [pendingDelete, ui.presets]);

  const presetsLabel = t('widgets.generate.presets');
  // Tooltip and accessible name carry the visible preset label (label-in-name).
  const triggerLabel = activePreset ? `${presetsLabel} · ${activePreset.label}` : presetsLabel;

  return (
    <>
      <Popover.Root
        ids={triggerIds}
        lazyMount
        open={isOpen}
        positioning={POPOVER_POSITIONING}
        unmountOnExit
        onOpenChange={handleOpenChange}
      >
        <Tooltip content={triggerLabel} ids={triggerIds}>
          <Popover.Trigger asChild>
            <IconButton
              aria-label={triggerLabel}
              color="fg.muted"
              px={activePreset ? '1' : '0'}
              size="2xs"
              variant="ghost"
              w={activePreset ? 'auto' : '6'}
            >
              <Icon as={BookmarkIcon} boxSize="3.5" />
              {activePreset ? (
                <Text as="span" fontSize="2xs" maxW="8rem" truncate>
                  {activePreset.label}
                </Text>
              ) : null}
            </IconButton>
          </Popover.Trigger>
        </Tooltip>
        <Portal>
          <Popover.Positioner>
            <PopoverContent w="22rem">
              <Popover.Body p="2.5">
                <Stack gap="2">
                  <PromptPanelHeader label={presetsLabel}>
                    <Button
                      disabled={!canSave}
                      h={PANEL_HEADER_CONTROL_HEIGHT}
                      size="2xs"
                      variant="ghost"
                      onClick={openSaveDialog}
                    >
                      <PlusIcon />
                      {t('widgets.generate.savePresetTitle')}
                    </Button>
                  </PromptPanelHeader>
                  {presets.length >= SEARCH_VISIBLE_MIN_PRESETS ? (
                    <InputGroup startElement={SEARCH_START_ELEMENT}>
                      <Input
                        aria-label={t('widgets.generate.searchPresets')}
                        placeholder={t('widgets.generate.searchPresets')}
                        size="xs"
                        value={searchTerm}
                        onChange={handleSearchChange}
                      />
                    </InputGroup>
                  ) : null}
                  {presets.length === 0 ? (
                    <Text color="fg.subtle" fontSize="xs">
                      {t('widgets.generate.presetsEmpty')}
                    </Text>
                  ) : filteredPresets.length === 0 ? (
                    <Text color="fg.subtle" fontSize="xs">
                      {t('widgets.generate.presetsNoMatches')}
                    </Text>
                  ) : (
                    <Scrollable maxH="16rem">
                      <Stack gap="0.5" separator={PRESET_ROW_SEPARATOR}>
                        {filteredPresets.map((preset) => (
                          <PresetRow
                            key={preset.id}
                            isActive={preset.id === activePreset?.id}
                            preset={preset}
                            onApply={applyPreset}
                            onDelete={setPendingDelete}
                            onRename={openRenameDialog}
                          />
                        ))}
                      </Stack>
                    </Scrollable>
                  )}
                </Stack>
              </Popover.Body>
            </PopoverContent>
          </Popover.Positioner>
        </Portal>
      </Popover.Root>
      {dialog ? (
        <RenameDialog
          initialName={dialog.mode === 'rename' ? dialog.preset.label : ''}
          isOpen
          label={t('widgets.generate.presetName')}
          submitLabel={dialog.mode === 'rename' ? t('common.rename') : t('widgets.generate.savePresetAction')}
          title={
            dialog.mode === 'rename' ? t('widgets.generate.renamePresetTitle') : t('widgets.generate.savePresetTitle')
          }
          onClose={closeDialog}
          onSubmit={handleDialogSubmit}
        />
      ) : null}
      <ConfirmDialog
        body={t('widgets.generate.presetDeleteBody', { name: pendingDelete?.label ?? '' })}
        confirmLabel={t('common.delete')}
        isOpen={pendingDelete !== null}
        title={t('widgets.generate.presetDeleteTitle')}
        onClose={cancelDelete}
        onConfirm={confirmDelete}
      />
    </>
  );
};
