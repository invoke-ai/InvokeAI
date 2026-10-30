import type { GenerateSettings } from '@features/generation/core/types';
import type { ChangeEvent } from 'react';

import { Checkbox, HStack, Icon, Input, InputGroup, Popover, Portal, Separator, Stack, Text } from '@chakra-ui/react';
import {
  getGenerateModelSelectionResult,
  isSupportedGenerateModel,
} from '@features/generation/core/baseGenerationPolicies';
import { GENERATE_TOOL_MODEL_PICK_KEYS, normalizeGenerateSettings } from '@features/generation/core/settings';
import { resolveGenerateWidgetValues } from '@features/generation/settings';
import { useExitRetainedValue } from '@platform/react/useExitRetainedValue';
import { Button, IconButton, PopoverContent, RenameDialog, Row, Scrollable, Tooltip } from '@platform/ui';
import { BookmarkIcon, CheckIcon, PencilIcon, PlusIcon, SearchIcon, Trash2Icon } from 'lucide-react';
import { useCallback, useId, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

import type { GeneratePresetRecord } from './GenerationUiContext';

import { flushGenerateDrafts } from './generateDraftRegistry';
import { getGenerateFormCommitPatch } from './generateFormViewModel';
import { offerGenerateUndo } from './generateUndo';
import { useGenerateValues, useGenerationUi } from './GenerationUiContext';
import { describeModelSwitchChanges } from './modelSelectionNotice';
import { PANEL_HEADER_CONTROL_HEIGHT, PromptPanelHeader } from './promptFields/PromptPanelHeader';

const POPOVER_POSITIONING = { placement: 'bottom-end' } as const;
const SEARCH_START_ELEMENT = <Icon as={SearchIcon} boxSize="3" color="fg.subtle" />;
// The rows sit on the popover's `bg.muted` surface; hover one surface step up.
const PRESET_ROW_HOVER_PROPS = { bg: 'bg.hover' };
const PRESET_ROW_SEPARATOR = <Separator borderColor="border.subtle" />;
/** Below this many presets, a search box is more furniture than help. */
const SEARCH_VISIBLE_MIN_PRESETS = 6;

type PresetDialogSubject = { mode: 'save' } | { mode: 'rename'; preset: GeneratePresetRecord };

/** What "include prompts" covers: the prompt text and the template applied to it, with their panel arrangement. */
const PRESET_PROMPT_KEYS = [
  'negativePrompt',
  'negativePromptEnabled',
  'negativePromptHeightPx',
  'positivePrompt',
  'positivePromptHeightPx',
  'promptTemplate',
  'promptTemplateViewMode',
] as const satisfies readonly (keyof GenerateSettings)[];

/** Settings newer than some saved presets: a preset without one keeps the current value instead of the default. */
const PRESET_NEWER_KEYS = ['dynamicPromptsEnabled'] as const satisfies readonly (keyof GenerateSettings)[];

/**
 * A preset's settings; one saved without prompts keeps the current prompts, one saved before a newer setting keeps the
 * current value of it, and everything else comes from the preset.
 */
const getPresetSettings = (current: object, preset: GeneratePresetRecord): GenerateSettings | null => {
  const currentValues = current as Record<string, unknown>;

  return normalizeGenerateSettings({
    ...Object.fromEntries([...PRESET_PROMPT_KEYS, ...PRESET_NEWER_KEYS].map((key) => [key, currentValues[key]])),
    ...preset.values,
  });
};

/** The keys a preset leaves as they are: the prompts when saved without them, and settings newer than it. */
const getPresetKeptKeys = (preset: GeneratePresetRecord): readonly string[] => [
  ...('positivePrompt' in preset.values ? [] : PRESET_PROMPT_KEYS),
  ...PRESET_NEWER_KEYS.filter((key) => !(key in preset.values)),
];

/** Ignore presentation and non-applied keys, including seed unless its mode is fixed, and the keys a preset keeps. */
const getPresetComparisonKey = (settings: GenerateSettings, keptKeys: readonly string[] = []): string => {
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

  for (const key of keptKeys) {
    delete comparable[key];
  }

  return JSON.stringify(comparable);
};

/**
 * A preset's comparison key, read once per preset: it leaves out the keys the preset keeps (prompts it was saved
 * without, settings newer than it), which match whatever is current. Placeholder prompts let it normalize on its own.
 */
const getStoredPresetKey = (preset: GeneratePresetRecord): { key: string; keptKeys: readonly string[] } | null => {
  const withoutPrompts = !('positivePrompt' in preset.values);
  const keptKeys = getPresetKeptKeys(preset);
  const settings = normalizeGenerateSettings(
    withoutPrompts ? { negativePrompt: '', positivePrompt: '', ...preset.values } : preset.values
  );

  return settings ? { key: getPresetComparisonKey(settings, keptKeys), keptKeys } : null;
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
          <Text flex="1" fontSize="md" fontWeight={isActive ? '600' : undefined} minW="0" textAlign="start" truncate>
            {preset.label}
          </Text>

          {isActive ? <Icon as={CheckIcon} boxSize="3.5" color="accent.fg" flexShrink={0} /> : null}
        </button>
      </Row>
      <Tooltip content={t('common.rename')}>
        <IconButton
          aria-label={t('widgets.generate.renamePresetNamed', { name: preset.label })}
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
          data-preset-delete={preset.id}
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
  const contentRef = useRef<HTMLDivElement>(null);
  const [dialogSubject, setDialogSubject] = useState<PresetDialogSubject | null>(null);
  const dialog = useExitRetainedValue(dialogSubject);
  // Tooltip and Popover must share the trigger ID to preserve anchoring.
  const triggerId = useId();
  const triggerIds = useMemo(() => ({ trigger: triggerId }), [triggerId]);
  // The popover that opened a dialog may have closed; focus returns to its trigger, not the page.
  const getTriggerElement = useCallback(() => document.getElementById(triggerId), [triggerId]);

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

  // Serialized equality requires canonical key order. Presets normalize once per catalog change, not per edit.
  const presetKeys = useMemo(() => presets.map(getStoredPresetKey), [presets]);
  const activePreset = useMemo(() => {
    if (!settings) {
      return null;
    }

    // Presets keep only a few distinct sets of keys, so the current settings are keyed once per set.
    const currentKeys = new Map<string, string>();
    const getCurrentKey = (keptKeys: readonly string[]) => {
      const signature = keptKeys.join();
      const known = currentKeys.get(signature);

      if (known !== undefined) {
        return known;
      }

      const key = getPresetComparisonKey(settings, keptKeys);

      currentKeys.set(signature, key);
      return key;
    };
    const index = presetKeys.findIndex((stored) => stored !== null && stored.key === getCurrentKey(stored.keptKeys));

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
      // Flush so Undo restores what the user saw, not a debounce behind it.
      flushGenerateDrafts();
      const before = ui.generateValues.getSnapshot();
      const normalized = getPresetSettings(before, record);

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

      const patch = getGenerateFormCommitPatch(resolved.values);
      // What the preset holds that its model cannot use now, told as a switch to that model would tell it.
      const message = describeModelSwitchChanges(result, t, i18n.resolvedLanguage).join(' ');

      ui.settings.patchGenerateSettings(patch, projectId);
      offerGenerateUndo({
        after: { ...before, ...patch },
        before,
        message,
        projectId,
        slot: 'settings',
        t,
        title: t('widgets.generate.presetApplied', { name: record.label }),
        ui,
      });
      setIsOpen(false);
    },
    [i18n.resolvedLanguage, models, projectId, supportedModels, t, ui]
  );

  const [includePrompts, setIncludePrompts] = useState(ui.presets.includePrompts);
  const openSaveDialog = useCallback(() => {
    setIncludePrompts(ui.presets.includePrompts);
    setDialogSubject({ mode: 'save' });
  }, [ui.presets.includePrompts]);
  const handleIncludePromptsChange = useCallback(
    (event: { checked: boolean | 'indeterminate' }) => setIncludePrompts(event.checked === true),
    []
  );
  const openRenameDialog = useCallback(
    (preset: GeneratePresetRecord) => setDialogSubject({ mode: 'rename', preset }),
    []
  );
  const closeDialog = useCallback(() => setDialogSubject(null), []);
  const handleDialogSubmit = useCallback(
    (label: string) => {
      if (dialogSubject?.mode === 'rename') {
        ui.presets.rename(dialogSubject.preset.id, label);
        return;
      }

      // Flush so the snapshot carries what the user sees, not a debounce behind it.
      flushGenerateDrafts();
      const snapshot = normalizeGenerateSettings(ui.generateValues.getSnapshot());

      if (!snapshot) {
        return;
      }

      const values: Record<string, unknown> = { ...snapshot };

      if (!includePrompts) {
        for (const key of PRESET_PROMPT_KEYS) {
          delete values[key];
        }
      }

      ui.presets.save(label, values);

      if (includePrompts !== ui.presets.includePrompts) {
        ui.presets.setIncludePrompts(includePrompts);
      }
    },
    [dialogSubject, includePrompts, ui.generateValues, ui.presets]
  );

  // Deleting is undoable, so it asks nothing. The active preset is derived from the settings, which a delete leaves
  // alone: Undo brings the preset back as active if the settings still match it.
  const deletePreset = useCallback(
    (preset: GeneratePresetRecord) => {
      const index = ui.presets.presets.findIndex((candidate) => candidate.id === preset.id);
      const { restore } = ui.presets;
      // Focus moves before the row leaves, to the next row's delete button (the previous one's after the last row), so
      // deleting several presets needs no Tab back from the top; with no row left, to the popover itself.
      const shownIndex = filteredPresets.findIndex((candidate) => candidate.id === preset.id);
      const neighbour = filteredPresets[shownIndex + 1] ?? filteredPresets[shownIndex - 1];
      const content = contentRef.current;
      const neighbourDelete = neighbour
        ? content?.querySelector<HTMLElement>(`[data-preset-delete="${CSS.escape(neighbour.id)}"]`)
        : null;

      (neighbourDelete ?? content)?.focus();
      ui.presets.remove(preset.id);
      ui.notifications.undoable(t('widgets.generate.presetDeleted', { name: preset.label }), {
        onUndo: () => restore(preset, index),
      });
    },
    [filteredPresets, t, ui.notifications, ui.presets]
  );

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
              size="sm"
              variant="ghost"
              w={activePreset ? 'auto' : '6'}
            >
              <Icon as={BookmarkIcon} boxSize="3.5" />
              {activePreset ? (
                <Text as="span" fontSize="xs" maxW="8rem" truncate>
                  {activePreset.label}
                </Text>
              ) : null}
            </IconButton>
          </Popover.Trigger>
        </Tooltip>
        <Portal>
          <Popover.Positioner>
            <PopoverContent ref={contentRef} w="22rem">
              <Popover.Body p="2.5">
                <Stack gap="2">
                  <PromptPanelHeader label={presetsLabel}>
                    <Button
                      disabled={!canSave}
                      h={PANEL_HEADER_CONTROL_HEIGHT}
                      size="sm"
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
                        value={searchTerm}
                        onChange={handleSearchChange}
                      />
                    </InputGroup>
                  ) : null}
                  {presets.length === 0 ? (
                    <Text color="fg.subtle" fontSize="md">
                      {t('widgets.generate.presetsEmpty')}
                    </Text>
                  ) : filteredPresets.length === 0 ? (
                    <Text color="fg.subtle" fontSize="md">
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
                            onDelete={deletePreset}
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
      {dialog.value ? (
        <RenameDialog
          key={dialog.generation}
          finalFocusEl={getTriggerElement}
          initialName={dialog.value.mode === 'rename' ? dialog.value.preset.label : ''}
          isOpen={dialog.isOpen}
          label={t('widgets.generate.presetName')}
          submitLabel={dialog.value.mode === 'rename' ? t('common.rename') : t('widgets.generate.savePresetAction')}
          title={
            dialog.value.mode === 'rename'
              ? t('widgets.generate.renamePresetTitle')
              : t('widgets.generate.savePresetTitle')
          }
          onClose={closeDialog}
          onExitComplete={dialog.release}
          onSubmit={handleDialogSubmit}
        >
          {dialog.value.mode === 'save' ? (
            <Checkbox.Root checked={includePrompts} size="sm" onCheckedChange={handleIncludePromptsChange}>
              <Checkbox.HiddenInput />
              <Checkbox.Control>
                <Checkbox.Indicator />
              </Checkbox.Control>
              <Checkbox.Label>{t('widgets.generate.presetIncludePrompts')}</Checkbox.Label>
            </Checkbox.Root>
          ) : null}
        </RenameDialog>
      ) : null}
    </>
  );
};
