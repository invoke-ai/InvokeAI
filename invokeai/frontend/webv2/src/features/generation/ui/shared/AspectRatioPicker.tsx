/* oxlint-disable react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop, react-perf/jsx-no-jsx-as-prop */
import type {
  AspectRatioPreset,
  AspectRatioPresetDraft,
  AspectRatioPresetsPort,
  AspectRatioTerms,
} from '@features/generation/core/aspectRatioPresets';
import type { AspectRatioId } from '@features/generation/core/types';
import type { FormEvent, KeyboardEvent, ReactNode } from 'react';

import {
  Flex,
  Grid,
  HStack,
  Icon,
  Input,
  Popover,
  Portal,
  Separator,
  Stack,
  Text,
  type TextProps,
  useFieldContext,
} from '@chakra-ui/react';
import {
  formatAspectRatio,
  formatAspectRatioTerms,
  getAspectRatioPresetRatio,
  getAspectRatioTerms,
  getAspectRatioTermsError,
  getLockedAspectRatioSettings,
  matchAspectRatio,
  MAX_ASPECT_RATIO_PRESET_LABEL_LENGTH,
  MAX_ASPECT_RATIO_PRESETS,
} from '@features/generation/core/aspectRatioPresets';
import { ASPECT_RATIO_MAP, ASPECT_RATIO_OPTIONS } from '@features/generation/core/settings';
import { Button, IconButton } from '@platform/ui/Button';
import { useFieldLabelId } from '@platform/ui/Field';
import { PopoverContent } from '@platform/ui/Popover';
import { Row } from '@platform/ui/Row';
import { Scrollable } from '@platform/ui/Scrollable';
import { ClippedTooltip, Tooltip } from '@platform/ui/Tooltip';
import { dropdownContent, dropdownGroupLabel, formControlInteraction } from '@theme/recipes';
import { CheckIcon, ChevronDownIcon, LockIcon, LockOpenIcon, PencilIcon, PlusIcon, Trash2Icon } from 'lucide-react';
import { useCallback, useId, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { AspectRatioPreview } from './AspectRatioPreview';

type ControlSize = 'md' | 'lg';

interface Size {
  width: number;
  height: number;
}

interface AspectRatioPickerCommonProps {
  disabled?: boolean;
  /** Shown on the trigger in place of the chosen ratio, e.g. the media that decides it while disabled. */
  valueText?: ReactNode;
}

/** Built-ins, Free, and the account's saved ratios with their add and edit form. */
export interface AspectRatioPickerFullProps extends AspectRatioPickerCommonProps {
  /** The live size: the Free thumbnail's shape and the Add ratio form's prefill. */
  currentSize: Size;
  /** The ratio the lock holds, or null while unlocked (Free). */
  lockedRatio: number | null;
  presets: AspectRatioPresetsPort;
  /** The size choosing `ratio` would set, shown beside each saved ratio. */
  getSizeForRatio: (ratio: number) => Size;
  /** A ratio to lock, or null for Free. Not called for the ratio already chosen. */
  onChange: (ratio: number | null) => void;
}

/** The built-in ratio grid only: no Free and no saved ratios. */
export interface AspectRatioPickerFixedProps extends AspectRatioPickerCommonProps {
  currentSize?: never;
  lockedRatio: number;
  presets?: never;
  getSizeForRatio?: never;
  /** A built-in's ratio. Not called for the ratio already chosen. */
  onChange: (ratio: number) => void;
}

export type AspectRatioPickerProps = AspectRatioPickerFullProps | AspectRatioPickerFixedProps;

type FormSubject = { mode: 'add' } | { mode: 'edit'; preset: AspectRatioPreset };

const POSITIONING = { placement: 'bottom-start' } as const;
const GRID_COLUMNS = 5;
// A gutter for the overlay scrollbar, which a long list would otherwise draw over the Delete column.
const SAVED_LIST_CONTENT_PROPS = { pe: '2.5' } as const;
// A literal of references, not a spread: a module-scope property read would make this barrel-exported module impure.
const TRIGGER_CSS = [formControlInteraction, { _expanded: { borderColor: 'accent.solid' } }];
const TILE_CSS = { flexDirection: 'column', gap: '0.5', justifyContent: 'center', px: '0.5', py: '1' } as const;
/** Edit and delete stay in the tab order; they show on hover and whenever focus is in the row (or with no hover). */
const PRESET_ROW_CSS = {
  '& [data-row-action]': { opacity: 0, transitionDuration: 'fast', transitionProperty: 'opacity' },
  '&:hover [data-row-action], &:focus-within [data-row-action]': { opacity: 1 },
  '@media (hover: none)': { '& [data-row-action]': { opacity: 1 } },
} as const;

const formatSize = (size: Size): string => `${Math.round(size.width)}×${Math.round(size.height)}`;

/** Text that truncates, with its whole text in a tooltip while it is cut short. */
const ClippedText = ({ children, ...props }: TextProps & { children: string }) => {
  const ref = useRef<HTMLParagraphElement>(null);
  const isClipped = useCallback(() => !!ref.current && ref.current.scrollWidth > ref.current.clientWidth, []);

  return (
    <ClippedTooltip content={children} isClipped={isClipped}>
      <Text ref={ref} as="span" minW="0" truncate {...props}>
        {children}
      </Text>
    </ClippedTooltip>
  );
};

/**
 * Digits with decimals after either separator; a trailing separator (`5.`) reads as the whole number. Nothing else
 * `Number` reads (`0x10`, `1e1`), and no thousands separator: `1,000` reads as a decimal with too many places.
 */
const TERM_PATTERN = /^\d*(?:[.,](\d*))?$/;
/** The precision a term keeps: saved terms come back to the edit form at two decimals. */
const MAX_TERM_DECIMALS = 2;

type TermError = 'number' | 'decimals';

/** A typed term above 0, or why it is not one: blank, unreadable or 0 (`number`), or too precise (`decimals`). */
const readTerm = (value: string): number | TermError => {
  const trimmed = value.trim();
  const match = trimmed === '' ? null : TERM_PATTERN.exec(trimmed);
  const term = match ? Number(trimmed.replace(',', '.')) : Number.NaN;

  if (!match || !Number.isFinite(term) || term <= 0) {
    return 'number';
  }

  return (match[1]?.length ?? 0) > MAX_TERM_DECIMALS ? 'decimals' : term;
};

/** Focus targets are keyed, so a control that mounts after the change that needs it can still take focus. */
const useDeferredFocus = () => {
  const pendingRef = useRef<string | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);

  const focus = useCallback((key: string) => {
    const target = containerRef.current?.querySelector<HTMLElement>(`[data-focus-key="${CSS.escape(key)}"]`);

    if (target) {
      pendingRef.current = null;
      target.focus();
    } else {
      pendingRef.current = key;
    }
  }, []);

  // Stable, so React calls it only as a target mounts: the moment a pending focus can land on it.
  const registerTarget = useCallback((element: HTMLElement | null) => {
    if (element && element.dataset.focusKey === pendingRef.current) {
      pendingRef.current = null;
      element.focus();
    }
  }, []);

  return { containerRef, focus, registerTarget };
};

type DeferredFocus = ReturnType<typeof useDeferredFocus>;

const PresetForm = ({
  getSizeForRatio,
  registerFocusTarget,
  initial,
  submitLabel,
  title,
  onCancel,
  onSubmit,
}: {
  /** Names the size the entered ratio would set, on the line under the terms while no error shows. */
  getSizeForRatio: (ratio: number) => Size;
  registerFocusTarget: DeferredFocus['registerTarget'];
  initial: { label: string; terms: AspectRatioTerms };
  submitLabel: string;
  title: string;
  onCancel: () => void;
  onSubmit: (draft: AspectRatioPresetDraft) => void;
}) => {
  const { i18n, t } = useTranslation();
  const nameErrorId = useId();
  const termsErrorId = useId();
  const hintId = useId();
  const formatTerm = (term: number) =>
    new Intl.NumberFormat(i18n.resolvedLanguage, { maximumFractionDigits: 2, useGrouping: false }).format(term);
  const [label, setLabel] = useState(initial.label);
  const [width, setWidth] = useState(() => formatTerm(initial.terms.width));
  const [height, setHeight] = useState(() => formatTerm(initial.terms.height));
  // Errors wait for the first submit, then follow every edit.
  const [hasSubmitted, setHasSubmitted] = useState(false);
  const nameError = label.trim() === '' ? t('widgets.generate.aspectRatioNameRequired') : null;
  const widthTerm = readTerm(width);
  const heightTerm = readTerm(height);
  // Each term is checked on its own, so only the offending field is marked; the range is the pair's.
  const termsErrorKind =
    widthTerm === 'number' || heightTerm === 'number'
      ? 'number'
      : widthTerm === 'decimals' || heightTerm === 'decimals'
        ? 'decimals'
        : getAspectRatioTermsError(widthTerm, heightTerm);
  const termsError =
    termsErrorKind === 'decimals'
      ? t('widgets.generate.aspectRatioTermDecimals')
      : termsErrorKind === 'range'
        ? t('widgets.generate.aspectRatioOutOfRange')
        : termsErrorKind
          ? t('widgets.generate.aspectRatioTermsInvalid')
          : null;
  // Follows the terms as they are typed; a dashed square until they make a ratio this build accepts.
  const previewRatio = termsErrorKind === null ? (widthTerm as number) / (heightTerm as number) : null;
  const showsTermsError = hasSubmitted && termsError !== null;
  // Marked only where the shown message applies: a term with another error waits until this one is fixed.
  const isTermInvalid = (term: number | TermError) =>
    showsTermsError && (term === termsErrorKind || termsErrorKind === 'range');
  const isWidthInvalid = isTermInvalid(widthTerm);
  const isHeightInvalid = isTermInvalid(heightTerm);
  const hintText =
    previewRatio === null
      ? ''
      : t('widgets.generate.aspectRatioWouldSetSize', { size: formatSize(getSizeForRatio(previewRatio)) });
  const describeTerm = (isInvalid: boolean) =>
    isInvalid ? termsErrorId : !showsTermsError && hintText ? hintId : undefined;

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    // The popover is portalled, so a submit would otherwise reach forms around the picker in the React tree.
    event.stopPropagation();
    setHasSubmitted(true);

    if (nameError || termsError) {
      // The name, else the first term the message is about; both are for a ratio out of range.
      const target = nameError
        ? '[data-focus-key="form-name"]'
        : widthTerm === termsErrorKind || termsErrorKind === 'range'
          ? '[data-term="width"]'
          : '[data-term="height"]';

      event.currentTarget.querySelector<HTMLElement>(target)?.focus();
      return;
    }

    onSubmit({ height: heightTerm as number, label: label.trim(), width: widthTerm as number });
  };

  return (
    <Stack asChild gap="2" py="1">
      <form aria-label={title} noValidate onSubmit={handleSubmit}>
        {/* A square preview well as tall as the fields beside it (name, terms and the line under them: 88px), at a fixed size
            so a name error's extra line moves nothing sideways. The line under the terms always holds the hint or the
            error, so neither the shape nor a terms error changes the form's height. */}
        <HStack alignItems="flex-start" gap="2">
          <Flex
            alignItems="center"
            borderColor="border.muted"
            borderWidth="1px"
            boxSize="5.5rem"
            data-preview-well=""
            flexShrink={0}
            justifyContent="center"
            p="1.5"
            rounded="l2"
          >
            <AspectRatioPreview boxSize="full" ratio={previewRatio} />
          </Flex>
          <Stack flex="1" gap="2" minW="0">
            <Stack gap="1">
              <Input
                ref={registerFocusTarget}
                aria-describedby={hasSubmitted && nameError ? nameErrorId : undefined}
                aria-invalid={(hasSubmitted && nameError !== null) || undefined}
                aria-label={t('widgets.generate.aspectRatioName')}
                autoComplete="off"
                data-focus-key="form-name"
                maxLength={MAX_ASPECT_RATIO_PRESET_LABEL_LENGTH}
                placeholder={t('widgets.generate.aspectRatioName')}
                value={label}
                onChange={(event) => setLabel(event.currentTarget.value)}
              />
              {hasSubmitted && nameError ? (
                <Text color="fg.error" fontSize="xs" id={nameErrorId} role="alert">
                  {nameError}
                </Text>
              ) : null}
            </Stack>
            <Stack gap="1">
              <HStack alignItems="center" gap="1.5">
                <Input
                  aria-describedby={describeTerm(isWidthInvalid)}
                  aria-invalid={isWidthInvalid || undefined}
                  aria-label={t('widgets.generate.aspectRatioWidthTerm')}
                  data-term="width"
                  flex="1"
                  inputMode="decimal"
                  minW="0"
                  textAlign="end"
                  value={width}
                  onChange={(event) => setWidth(event.currentTarget.value)}
                />
                <Text aria-hidden color="fg.muted" fontSize="md">
                  :
                </Text>
                <Input
                  aria-describedby={describeTerm(isHeightInvalid)}
                  aria-invalid={isHeightInvalid || undefined}
                  aria-label={t('widgets.generate.aspectRatioHeightTerm')}
                  data-term="height"
                  flex="1"
                  inputMode="decimal"
                  minW="0"
                  value={height}
                  onChange={(event) => setHeight(event.currentTarget.value)}
                />
              </HStack>
              {showsTermsError ? (
                <Text color="fg.error" fontSize="xs" id={termsErrorId} role="alert">
                  {termsError}
                </Text>
              ) : (
                // A blank line keeps its height while the terms make no ratio to describe.
                <Text color="fg.muted" fontSize="xs" id={hintId} minH="1lh">
                  {hintText}
                </Text>
              )}
            </Stack>
          </Stack>
        </HStack>
        <HStack gap="1.5" justifyContent="end">
          <Button type="button" variant="ghost" onClick={onCancel}>
            {t('common.cancel')}
          </Button>
          <Button type="submit">{submitLabel}</Button>
        </HStack>
      </form>
    </Stack>
  );
};

const PresetRow = ({
  registerFocusTarget,
  isCurrent,
  preset,
  ratioText,
  sizeText,
  onChoose,
  onDelete,
  onEdit,
}: {
  registerFocusTarget: DeferredFocus['registerTarget'];
  isCurrent: boolean;
  preset: AspectRatioPreset;
  ratioText: string;
  sizeText: string;
  onChoose: () => void;
  onDelete: () => void;
  onEdit: () => void;
}) => {
  const { t } = useTranslation();
  const names = { name: preset.label, ratio: ratioText };

  return (
    <HStack css={PRESET_ROW_CSS} gap="0.5">
      <Row active={isCurrent ? 'emphasized' : 'none'} asChild flex="1" gap="1.5" minH="10" minW="0" px="1.5" py="1">
        <button
          ref={registerFocusTarget}
          data-focus-key={`choose:${preset.id}`}
          aria-current={isCurrent || undefined}
          aria-label={t('widgets.generate.aspectRatioPresetOption', { ...names, size: sizeText })}
          data-picker-current={isCurrent || undefined}
          type="button"
          onClick={onChoose}
        >
          <AspectRatioPreview boxSize="5" ratio={getAspectRatioPresetRatio(preset)} />
          <Stack as="span" flex="1" gap="0" minW="0" textAlign="start">
            <ClippedText fontWeight={isCurrent ? '600' : undefined} textStyle="md">
              {preset.label}
            </ClippedText>
            <HStack as="span" gap="1.5" fontVariantNumeric="tabular-nums" textStyle="sm">
              <Text as="span" color="fg.muted" flexShrink={0}>
                {ratioText}
              </Text>
              <Text as="span" color="fg.subtle" flexShrink={0}>
                {sizeText}
              </Text>
            </HStack>
          </Stack>
          <Icon as={CheckIcon} aria-hidden boxSize="3" color="accent.fg" flexShrink={0} opacity={isCurrent ? 1 : 0} />
        </button>
      </Row>
      <IconButton
        ref={registerFocusTarget}
        data-focus-key={`edit:${preset.id}`}
        aria-label={t('widgets.generate.aspectRatioEditNamed', names)}
        data-row-action=""
        tooltip={t('widgets.generate.aspectRatioEdit')}
        variant="ghost"
        onClick={onEdit}
      >
        <PencilIcon />
      </IconButton>
      <IconButton
        ref={registerFocusTarget}
        data-focus-key={`delete:${preset.id}`}
        aria-label={t('widgets.generate.aspectRatioDeleteNamed', names)}
        colorPalette="red"
        data-row-action=""
        tooltip={t('common.delete')}
        variant="ghost"
        onClick={onDelete}
      >
        <Trash2Icon />
      </IconButton>
    </HStack>
  );
};

const GroupLabel = ({ children, id }: { children: ReactNode; id: string }) => (
  <Text css={dropdownGroupLabel} id={id} px="1">
    {children}
  </Text>
);

/**
 * The aspect ratio control: built-in ratios as a grid of shapes, the account's saved ratios as rows, and an inline
 * form to add or edit one. Choosing applies at once and closes; callers own what a ratio does to the size.
 * Without `presets` it offers the built-in grid alone.
 */
export const AspectRatioPicker = (props: AspectRatioPickerProps) => {
  const { disabled, lockedRatio, valueText: valueTextOverride } = props;
  // Like Select, an enclosing Field's label names the trigger and its disabled state applies; else a hidden label does.
  const field = useFieldContext();
  const fieldLabelId = useFieldLabelId();
  const full = props.presets ? props : null;
  const savedPresets = full?.presets.presets ?? [];
  const { i18n, t } = useTranslation();
  const locale = i18n.resolvedLanguage;
  const [isOpen, setIsOpen] = useState(false);
  const [form, setForm] = useState<FormSubject | null>(null);
  const { containerRef, focus: focusKey, registerTarget } = useDeferredFocus();
  const hiddenLabelId = useId();
  const labelId = fieldLabelId ?? hiddenLabelId;
  const valueId = useId();
  const builtInsId = useId();
  const savedId = useId();
  const tileRefs = useRef<(HTMLButtonElement | null)[]>([]);

  const currentRatio = full
    ? full.currentSize.height > 0
      ? full.currentSize.width / full.currentSize.height
      : 1
    : (lockedRatio ?? 1);
  const match = lockedRatio === null ? null : matchAspectRatio(lockedRatio, savedPresets);
  const builtIns = ASPECT_RATIO_OPTIONS.filter((id) => full || id !== 'Free').map((id) => ({
    id,
    ratio: id === 'Free' ? null : ASPECT_RATIO_MAP[id].ratio,
  }));
  // The built-in equal to the locked ratio is current even while a saved ratio names it, so choosing either closes.
  const lockedBuiltInId = lockedRatio === null ? 'Free' : getLockedAspectRatioSettings(lockedRatio).aspectRatioId;
  const currentTileIndex =
    lockedRatio === null
      ? 0
      : lockedBuiltInId === 'Free'
        ? -1
        : builtIns.findIndex((tile) => tile.id === lockedBuiltInId);
  const currentPresetId = match?.kind === 'preset' ? match.preset.id : null;
  const [focusedTile, setFocusedTile] = useState(Math.max(currentTileIndex, 0));

  const formatPresetRatio = (preset: AspectRatioTerms) => formatAspectRatioTerms(preset, locale);
  const freeLabel = t('widgets.generate.aspectRatioFree');
  const tileLabel = (id: AspectRatioId) => (id === 'Free' ? freeLabel : id);

  const valueText =
    match?.kind === 'preset' ? (
      <>
        <ClippedText>{match.preset.label}</ClippedText>{' '}
        <Text as="span" color="fg.muted" flexShrink={0}>
          {formatPresetRatio(match.preset)}
        </Text>
      </>
    ) : (
      <Text as="span" minW="0" truncate>
        {match?.kind === 'builtin'
          ? match.id
          : lockedRatio !== null
            ? formatAspectRatio(lockedRatio, locale)
            : freeLabel}
      </Text>
    );

  const handleOpenChange = ({ open }: { open: boolean }) => {
    setIsOpen(open);

    if (!open) {
      setForm(null);
    } else {
      setFocusedTile(Math.max(currentTileIndex, 0));
    }
  };

  const choose = (ratio: number | null, isCurrent: boolean) => {
    if (!isCurrent) {
      // Only Free is null, and the built-ins-only grid has no Free tile.
      (props.onChange as (ratio: number | null) => void)(ratio);
    }

    setIsOpen(false);
    setForm(null);
  };

  const openForm = (subject: FormSubject) => {
    setForm(subject);
    focusKey('form-name');
  };

  const closeForm = () => {
    if (!form) {
      return;
    }

    setForm(null);
    focusKey(form.mode === 'edit' ? `edit:${form.preset.id}` : 'add');
  };

  const submitForm = (draft: AspectRatioPresetDraft) => {
    if (!full) {
      return;
    }

    const { presets } = full;

    if (form?.mode === 'edit') {
      presets.update(form.preset.id, draft);
      setForm(null);
      focusKey(`choose:${form.preset.id}`);
      return;
    }

    const saved = presets.save(draft);
    // A full list saves nothing and shows no Add button; focus goes to its last row instead.
    const last = presets.presets.at(-1);

    setForm(null);
    focusKey(saved ? `choose:${saved.id}` : last ? `choose:${last.id}` : 'add');
  };

  // Deleting is undoable (the port reports it with Undo), so it asks nothing. Focus moves to the next row's delete,
  // the previous one's after the last row, or Add ratio once none are left.
  const deletePreset = (preset: AspectRatioPreset) => {
    if (!full) {
      return;
    }

    const { presets } = full;
    const index = presets.presets.findIndex((candidate) => candidate.id === preset.id);
    const neighbour = presets.presets[index + 1] ?? presets.presets[index - 1];

    focusKey(neighbour ? `delete:${neighbour.id}` : 'add');
    presets.remove(preset.id);
  };

  const handleTileKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    const last = builtIns.length - 1;
    const moves: Record<string, number> = {
      ArrowDown: Math.min(focusedTile + GRID_COLUMNS, last),
      ArrowLeft: Math.max(focusedTile - 1, 0),
      ArrowRight: Math.min(focusedTile + 1, last),
      ArrowUp: Math.max(focusedTile - GRID_COLUMNS, 0),
      End: last,
      Home: 0,
    };
    const next = moves[event.key];

    if (next === undefined) {
      return;
    }

    event.preventDefault();
    setFocusedTile(next);
    tileRefs.current[next]?.focus();
  };

  const getInitialFocus = () =>
    containerRef.current?.querySelector<HTMLElement>('[data-focus-key][data-picker-current]') ??
    containerRef.current?.querySelector<HTMLElement>('[data-picker-current]') ??
    tileRefs.current[0] ??
    null;

  const formFor = (subject: FormSubject, getSizeForRatio: (ratio: number) => Size) =>
    subject.mode === 'edit' ? (
      <PresetForm
        key={subject.preset.id}
        getSizeForRatio={getSizeForRatio}
        registerFocusTarget={registerTarget}
        initial={{ label: subject.preset.label, terms: subject.preset }}
        submitLabel={t('common.save')}
        title={t('widgets.generate.aspectRatioEdit')}
        onCancel={closeForm}
        onSubmit={submitForm}
      />
    ) : (
      <PresetForm
        getSizeForRatio={getSizeForRatio}
        registerFocusTarget={registerTarget}
        // A new ratio starts from the locked ratio (a snapped size is only close to it), else from the size itself.
        initial={{ label: '', terms: getAspectRatioTerms(lockedRatio ?? currentRatio) }}
        submitLabel={t('common.add')}
        title={t('widgets.generate.aspectRatioAdd')}
        onCancel={closeForm}
        onSubmit={submitForm}
      />
    );

  return (
    <Popover.Root
      // The field label's `for` points at its control, so the trigger takes that id.
      ids={fieldLabelId && field ? { trigger: field.ids.control } : undefined}
      initialFocusEl={getInitialFocus}
      lazyMount
      open={isOpen}
      positioning={POSITIONING}
      unmountOnExit
      // Escape inside the form cancels the form, not the popover.
      onEscapeKeyDown={(event) => {
        if (form) {
          event.preventDefault();
          closeForm();
        }
      }}
      onOpenChange={handleOpenChange}
    >
      {fieldLabelId ? null : (
        <Text hidden id={hiddenLabelId}>
          {t('widgets.generate.aspectRatio')}
        </Text>
      )}
      <Popover.Trigger asChild>
        <Button
          aria-labelledby={`${labelId} ${valueId}`}
          css={TRIGGER_CSS}
          disabled={disabled || field?.disabled}
          flex="1"
          justifyContent="space-between"
          minW="0"
          px="2"
          variant="outline"
          w="full"
        >
          <HStack as="span" gap="2" id={valueId} minW="0">
            {valueTextOverride ?? (
              <>
                <AspectRatioPreview boxSize="5" ratio={lockedRatio ?? currentRatio} />
                {valueText}
              </>
            )}
          </HStack>
          <Icon as={ChevronDownIcon} aria-hidden boxSize="3" color="fg.muted" flexShrink={0} />
        </Button>
      </Popover.Trigger>
      <Portal>
        <Popover.Positioner>
          <PopoverContent
            ref={containerRef}
            aria-labelledby={labelId}
            css={dropdownContent}
            maxW="calc(100vw - 1rem)"
            p="0"
            showArrow={false}
            w="21.5rem"
          >
            <Popover.Body p="1.5">
              <Stack gap="1.5">
                <GroupLabel id={builtInsId}>{t('widgets.generate.aspectRatioBuiltIns')}</GroupLabel>
                <Grid
                  aria-labelledby={builtInsId}
                  gap="0.5"
                  role="radiogroup"
                  templateColumns={`repeat(${GRID_COLUMNS}, minmax(0, 1fr))`}
                  onKeyDown={handleTileKeyDown}
                >
                  {builtIns.map((tile, index) => {
                    const isCurrent = index === currentTileIndex;

                    return (
                      <Row key={tile.id} active={isCurrent ? 'emphasized' : 'none'} asChild css={TILE_CSS}>
                        <button
                          ref={(element) => {
                            tileRefs.current[index] = element;
                          }}
                          aria-checked={isCurrent}
                          data-picker-current={isCurrent || undefined}
                          // Arrows move focus without choosing; Enter or Space chooses and closes.
                          role="radio"
                          tabIndex={index === focusedTile ? 0 : -1}
                          type="button"
                          onClick={() => choose(tile.ratio, isCurrent)}
                          onFocus={() => setFocusedTile(index)}
                        >
                          <AspectRatioPreview boxSize="6" ratio={tile.ratio ?? currentRatio} />
                          <Text
                            as="span"
                            color={isCurrent ? 'fg' : 'fg.muted'}
                            fontSize="xs"
                            fontVariantNumeric="tabular-nums"
                            fontWeight={isCurrent ? '600' : undefined}
                          >
                            {tileLabel(tile.id)}
                          </Text>
                        </button>
                      </Row>
                    );
                  })}
                </Grid>
                {full ? (
                  <>
                    <Separator borderColor="border.subtle" />
                    <GroupLabel id={savedId}>{t('widgets.generate.aspectRatioYourRatios')}</GroupLabel>
                    {full.presets.presets.length === 0 && form?.mode !== 'add' ? (
                      <Text color="fg.subtle" fontSize="sm" px="1">
                        {t('widgets.generate.aspectRatioYourRatiosEmpty')}
                      </Text>
                    ) : full.presets.presets.length > 0 ? (
                      <Scrollable contentProps={SAVED_LIST_CONTENT_PROPS} maxH="12rem">
                        <Stack aria-labelledby={savedId} gap="0.5" role="group">
                          {full.presets.presets.map((preset) =>
                            form?.mode === 'edit' && form.preset.id === preset.id ? (
                              formFor(form, full.getSizeForRatio)
                            ) : (
                              <PresetRow
                                key={preset.id}
                                registerFocusTarget={registerTarget}
                                isCurrent={preset.id === currentPresetId}
                                preset={preset}
                                ratioText={formatPresetRatio(preset)}
                                sizeText={formatSize(full.getSizeForRatio(getAspectRatioPresetRatio(preset)))}
                                onChoose={() =>
                                  choose(getAspectRatioPresetRatio(preset), preset.id === currentPresetId)
                                }
                                onDelete={() => deletePreset(preset)}
                                onEdit={() => openForm({ mode: 'edit', preset })}
                              />
                            )
                          )}
                        </Stack>
                      </Scrollable>
                    ) : null}
                    {form?.mode === 'add' ? (
                      formFor(form, full.getSizeForRatio)
                    ) : full.presets.isFull ? (
                      <Text color="fg.subtle" fontSize="sm" px="1">
                        {t('widgets.generate.aspectRatioListFull', { count: MAX_ASPECT_RATIO_PRESETS })}
                      </Text>
                    ) : (
                      <Button
                        ref={registerTarget}
                        data-focus-key="add"
                        justifyContent="start"
                        px="1.5"
                        variant="ghost"
                        onClick={() => openForm({ mode: 'add' })}
                      >
                        <PlusIcon />
                        {t('widgets.generate.aspectRatioAdd')}
                      </Button>
                    )}
                  </>
                ) : null}
              </Stack>
            </Popover.Body>
          </PopoverContent>
        </Popover.Positioner>
      </Portal>
    </Popover.Root>
  );
};

export interface AspectRatioLockButtonProps {
  isLocked: boolean;
  size?: 'sm' | ControlSize;
  onToggle: () => void;
}

/** Companion lock for {@link AspectRatioPicker}: a labelled icon button, not a bare dot. */
export const AspectRatioLockButton = ({ isLocked, size = 'md', onToggle }: AspectRatioLockButtonProps) => {
  const { t } = useTranslation();
  const label = isLocked ? t('widgets.generate.unlockAspectRatio') : t('widgets.generate.lockAspectRatio');

  return (
    <Tooltip content={label}>
      <IconButton aria-label={label} size={size} variant={isLocked ? 'solid' : 'outline'} onClick={onToggle}>
        {isLocked ? <LockIcon /> : <LockOpenIcon />}
      </IconButton>
    </Tooltip>
  );
};
