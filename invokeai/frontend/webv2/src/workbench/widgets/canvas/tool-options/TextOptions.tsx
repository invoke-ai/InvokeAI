import type { SelectValueChangeDetails } from '@chakra-ui/react';
import type { CanvasTextFontRef, TextToolOptions } from '@workbench/canvas-engine/api';
import type { TextSource } from '@workbench/widgets/canvas/textFontStyle';
import type { ToolFormProps, ToolPreviewProps } from '@workbench/widgets/canvas/tool-presentation/toolFormContracts';
import type { CSSProperties } from 'react';

import { Box, createListCollection, HStack, Spinner, Stack, Text } from '@chakra-ui/react';
import { fontsInfiniteQueryOptions, type FontAxis, type FontRecord } from '@features/fonts';
import { Button, IconButton, ToggleIconButton } from '@platform/ui/Button';
import { ColorPicker } from '@platform/ui/ColorPicker';
import { ScrubberField } from '@platform/ui/ScrubberField';
import { Select } from '@platform/ui/Select';
import { Tooltip } from '@platform/ui/Tooltip';
import { useInfiniteQuery } from '@tanstack/react-query';
import {
  DEFAULT_TEXT_OPTIONS,
  MAX_TEXT_FONT_SIZE,
  MIN_TEXT_FONT_SIZE,
  TEXT_FONT_FAMILIES,
  TEXT_FONT_WEIGHTS,
  getDocumentLayer,
} from '@workbench/canvas-engine/api';
import { useActiveColorCommands, useActiveColorPair } from '@workbench/widgets/canvas/color-system/useActiveColors';
import { useTextEditSession, useTextOptions } from '@workbench/widgets/canvas/engineStoreHooks';
import {
  TextFontReadiness,
  textFontKey,
  textFontVariationSettings,
  useResolvedTextFontFamily,
} from '@workbench/widgets/canvas/textFontStyle';
import { useSliderGesture } from '@workbench/widgets/canvas/tool-presentation/FormControls';
import { EditTargetChip, PropertyControlRow } from '@workbench/widgets/canvas/tool-presentation/PropertyPrimitives';
import { useColorSampler } from '@workbench/widgets/canvas/useColorSampler';
import { usePreparedCommit } from '@workbench/widgets/canvas/useStructuralCommit';
import { useActiveProjectSelector } from '@workbench/WorkbenchContext';
import { AlignCenterIcon, AlignLeftIcon, AlignRightIcon, RotateCcwIcon, SlidersHorizontalIcon } from 'lucide-react';
import { useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

type TextAlign = TextToolOptions['align'];

interface SelectedText {
  id: string;
  name: string;
  source: TextSource;
}

const SELECT_POSITIONING = { placement: 'bottom-start', sameWidth: false } as const;
const WEIGHT_TRIGGER_PROPS = { minW: '4.5rem', w: '4.5rem' } as const;

const ALIGN_ICONS: Record<TextAlign, typeof AlignLeftIcon> = {
  center: AlignCenterIcon,
  left: AlignLeftIcon,
  right: AlignRightIcon,
};

const ALIGN_LABEL_KEYS: Record<TextAlign, string> = {
  center: 'widgets.canvas.toolOptions.textAlignCenter',
  left: 'widgets.canvas.toolOptions.textAlignLeft',
  right: 'widgets.canvas.toolOptions.textAlignRight',
};

const ALIGN_VALUES: readonly TextAlign[] = ['left', 'center', 'right'];
const EMPTY_FONT_VARIATIONS: Readonly<Record<string, number>> = {};
const EMPTY_FONT_RECORDS: readonly FontRecord[] = [];

/** The log track covers the sizes text is actually set at; typing and keys still reach the document limits. */
const SIZE_TRACK_MIN_PX = 4;
const SIZE_TRACK_MAX_PX = 600;
const SPECIMEN_MAX_CHARS = 40;
const SPECIMEN_FALLBACK = 'Aa';
/** The canvas surround's checker, so any text color reads the way it does over a transparent document. */
const SPECIMEN_CHECKER =
  'conic-gradient({colors.bg.subtle} 25%, transparent 0 50%, {colors.bg.subtle} 0 75%, transparent 0)';

const formatLineHeight = (value: number): string => value.toFixed(1);

const clampTextSize = (value: number): number =>
  Math.min(MAX_TEXT_FONT_SIZE, Math.max(MIN_TEXT_FONT_SIZE, Math.round(value)));

const formatTextSize = (size: number): string => `${size}px`;

/** Whole pixels below 100, tens above: a log track's own step is sub-pixel at small sizes. */
export const textSizeKeyboardStep = (size: number, direction: -1 | 1): number =>
  size < 100 || (direction < 0 && size === 100) ? 1 : 10;

type FontChoiceGroup = 'builtin' | 'custom';

interface FontChoice {
  family: string;
  font?: FontRecord;
  group: FontChoiceGroup;
  label: string;
  ref?: CanvasTextFontRef;
  value: string;
}

const sameFontRef = (left: CanvasTextFontRef | undefined, right: CanvasTextFontRef | undefined): boolean =>
  left === right ||
  (left !== undefined &&
    right !== undefined &&
    left.id === right.id &&
    left.contentHash === right.contentHash &&
    left.family === right.family &&
    left.label === right.label);

const fontReferenceFromRecord = (font: FontRecord): CanvasTextFontRef => ({
  contentHash: font.contentHash,
  family: font.family,
  id: font.id,
  label: font.label,
});

const fontChoiceValue = (font: FontRecord | CanvasTextFontRef): string =>
  `custom:${font.id}:${font.contentHash}:${font.label}`;

const normalizeFontStyle = (style: string): NonNullable<TextSource['fontStyle']> =>
  style === 'italic' || style === 'oblique' ? style : 'normal';

/** Keeps v3 built-in sources free of optional defaults that force a v4 rewrite. */
export const canonicalizeSelectedTextSource = (source: TextSource): TextSource => {
  const next = { ...source };
  if (next.fontRef === undefined) {
    delete next.fontRef;
  }
  if (next.fontStyle === undefined || next.fontStyle === 'normal') {
    delete next.fontStyle;
  }
  if (!next.fontVariations || Object.keys(next.fontVariations).length === 0) {
    delete next.fontVariations;
  } else {
    next.fontVariations = { ...next.fontVariations };
  }
  return next;
};

const clampFontAxisValue = (value: number, axis: FontAxis): number =>
  Math.min(axis.maximum, Math.max(axis.minimum, Number.isFinite(value) ? value : axis.default));

/** Keeps variation maps aligned with the catalog's declared axes and exact coordinates. */
const fontVariationCoordinates = (
  font: FontRecord,
  coordinates: Readonly<Record<string, number>> = {}
): Record<string, number> =>
  Object.fromEntries(
    font.axes.map((axis) => [axis.tag, clampFontAxisValue(coordinates[axis.tag] ?? axis.default, axis)])
  );

/** Axes with a range to edit; a degenerate axis owns nothing. */
const usableAxes = (font: FontRecord | undefined): readonly FontAxis[] =>
  font?.axes.filter((axis) => axis.maximum > axis.minimum) ?? [];

/** An axis that replaces the plain Style or Weight select stays visible even when the font flags it hidden. */
const OWNING_AXIS_TAGS = new Set(['ital', 'slnt', 'wght']);

const axisStep = (axis: FontAxis): number => {
  const span = axis.maximum - axis.minimum;
  if (span <= 2) {
    return 0.01;
  }
  if (span <= 200) {
    return 0.1;
  }
  return 1;
};

/** The first line of the text being styled, or a neutral specimen when there is none yet. */
export const textSpecimen = (content: string | null): string => {
  const line =
    content
      ?.split('\n')
      .find((candidate) => candidate.trim().length > 0)
      ?.trim() ?? '';
  return line.length > 0 ? line.slice(0, SPECIMEN_MAX_CHARS) : SPECIMEN_FALLBACK;
};

const AlignButton = ({
  active,
  onSelect,
  value,
}: {
  active: boolean;
  onSelect: (value: TextAlign) => void;
  value: TextAlign;
}) => {
  const { t } = useTranslation();
  const Icon = ALIGN_ICONS[value];
  const onClick = useCallback(() => onSelect(value), [onSelect, value]);
  return (
    <Tooltip content={t(ALIGN_LABEL_KEYS[value])}>
      <IconButton
        aria-label={t(ALIGN_LABEL_KEYS[value])}
        aria-pressed={active}
        variant={active ? 'solid' : 'ghost'}
        onClick={onClick}
      >
        <Icon />
      </IconButton>
    </Tooltip>
  );
};

/** A text style edit: a patch, or one derived from the owner's style as it is when the edit is written. */
type TextEdit = Partial<TextSource> | ((style: TextToolOptions) => Partial<TextSource>);

/** The style an edit builds on: the owner's (live session or selected text) when there is one, else the defaults. */
const textStyleOf = (options: TextToolOptions, owner: TextSource | null): TextToolOptions => ({
  align: owner?.align ?? options.align,
  fontFamily: owner?.fontFamily ?? options.fontFamily,
  fontRef: owner ? owner.fontRef : options.fontRef,
  fontSize: owner?.fontSize ?? options.fontSize,
  fontStyle: (owner ? owner.fontStyle : options.fontStyle) ?? 'normal',
  fontVariations: (owner ? owner.fontVariations : options.fontVariations) ?? EMPTY_FONT_VARIATIONS,
  fontWeight: owner?.fontWeight ?? options.fontWeight,
  lineHeight: owner?.lineHeight ?? options.lineHeight,
});

/**
 * Prefer live text session, selected text, then defaults. Style edits update defaults and the current owner;
 * unowned color edits change the foreground pair.
 */
const useTextEditor = (engine: ToolFormProps['engine']) => {
  const { t } = useTranslation();
  const commitPrepared = usePreparedCommit(engine);
  const options = useTextOptions(engine);
  const pair = useActiveColorPair();
  const colorCommands = useActiveColorCommands();
  const session = useTextEditSession(engine);
  const selected = useActiveProjectSelector(
    (project): SelectedText | null => {
      const { document } = project.canvas;
      const layer = document.selectedLayerId ? getDocumentLayer(document, document.selectedLayerId) : undefined;
      return layer && layer.type === 'raster' && layer.source.type === 'text'
        ? { id: layer.id, name: layer.name, source: layer.source }
        : null;
    },
    (a, b) => a?.id === b?.id && a?.name === b?.name && a?.source === b?.source
  );
  const styleSource = session ? session.source : (selected?.source ?? null);
  const align = styleSource?.align ?? options.align;
  const fontFamily = styleSource?.fontFamily ?? options.fontFamily;
  const fontSize = styleSource?.fontSize ?? options.fontSize;
  const fontWeight = styleSource?.fontWeight ?? options.fontWeight;
  const fontRef = styleSource ? styleSource.fontRef : options.fontRef;
  const fontStyle = styleSource ? (styleSource.fontStyle ?? 'normal') : (options.fontStyle ?? 'normal');
  const fontVariations = styleSource
    ? (styleSource.fontVariations ?? EMPTY_FONT_VARIATIONS)
    : (options.fontVariations ?? EMPTY_FONT_VARIATIONS);
  const lineHeight = styleSource?.lineHeight ?? options.lineHeight;
  const color = styleSource?.color ?? pair.foreground;
  const content = styleSource?.content ?? null;
  const active = useMemo(
    () => ({ align, color, fontFamily, fontRef, fontSize, fontStyle, fontVariations, fontWeight, lineHeight }),
    [align, color, fontFamily, fontRef, fontSize, fontStyle, fontVariations, fontWeight, lineHeight]
  );
  const selectedId = selected?.id ?? null;
  // Scrubbers keep the handler a gesture started with, so everything an edit builds on is read when it is written:
  // an undo, hotkey or session change landing mid-gesture survives the release.
  const applyEdit = useCallback(
    (edit: TextEdit, commit: boolean) => {
      const liveSession = engine.interaction.get('textEditSession');
      const liveSelected = (): TextSource | null => {
        const layer = selectedId ? engine.document.model()?.getLayer(selectedId) : null;
        return layer?.type === 'raster' && layer.source.type === 'text' ? layer.source : null;
      };
      const style = textStyleOf(engine.interaction.get('textOptions'), liveSession?.source ?? liveSelected());
      const patch = typeof edit === 'function' ? edit(style) : edit;
      const { color: colorPatch, ...stylePatch } = patch;
      if (Object.keys(stylePatch).length > 0) {
        engine.interaction.set('textOptions', { ...style, ...stylePatch });
      }
      if (liveSession) {
        engine.layers.updateTextEditStyle(patch);
        return;
      }
      if (selectedId) {
        if (commit) {
          commitPrepared(t('widgets.canvas.toolOptions.textEdit'), (model) => {
            const layer = model.getLayer(selectedId);
            if (layer?.type !== 'raster' || layer.source.type !== 'text') {
              return { ids: [selectedId], status: 'missing' };
            }
            const after = canonicalizeSelectedTextSource({ ...layer.source, ...patch });
            return model.prepare({ id: selectedId, source: after, type: 'patch-source' });
          });
        }
        return;
      }
      if (colorPatch !== undefined) {
        colorCommands.setPairColor('foreground', colorPatch);
      }
    },
    [colorCommands, commitPrepared, engine, selectedId, t]
  );
  // Name the actual edit target: live session, selected text layer, or defaults.
  const sessionLayerName = useActiveProjectSelector((project): string | null => {
    if (!session?.layerId) {
      return null;
    }
    return getDocumentLayer(project.canvas.document, session.layerId)?.name ?? null;
  });
  const targetName = session ? (sessionLayerName ?? t('widgets.properties.target.newText')) : (selected?.name ?? null);
  return { active, applyEdit, content, targetName };
};

export const TextPreview = ({ engine }: ToolPreviewProps) => {
  const { active, content, targetName } = useTextEditor(engine);
  // The instanced face the canvas rasterizes with; the engine keeps the
  // session's, the document's and the defaults' faces active, so the load is shared.
  const face = useMemo((): TextSource => ({ ...active, content: '', type: 'text' }), [active]);
  const fontFamily = useResolvedTextFontFamily(engine.fonts, face);
  const style = useMemo(
    (): CSSProperties => ({
      color: active.color,
      fontFamily,
      fontStyle: active.fontStyle,
      fontVariationSettings: textFontVariationSettings(active) || 'normal',
      fontWeight: active.fontWeight,
      textAlign: active.align,
    }),
    [active, fontFamily]
  );
  return (
    <Stack bg="bg.inset" bgImage={SPECIMEN_CHECKER} bgSize="1rem 1rem" gap="1" px="3" py="2" rounded="sm">
      <TextFontReadiness key={textFontKey(face)} fonts={engine.fonts} source={face} />
      <EditTargetChip layerName={targetName} />
      <Text aria-hidden fontSize="2rem" lineHeight="1.25" minW="0" style={style} truncate>
        {textSpecimen(content)}
      </Text>
    </Stack>
  );
};

const AxisControl = ({
  activeVariations,
  applyEdit,
  axis,
}: {
  activeVariations: Readonly<Record<string, number>>;
  applyEdit: (edit: TextEdit, commit: boolean) => void;
  axis: FontAxis;
}) => {
  const committed = clampFontAxisValue(activeVariations[axis.tag] ?? axis.default, axis);
  const step = axisStep(axis);
  const precision = step < 0.1 ? 2 : step < 1 ? 1 : 0;
  const setValue = useCallback(
    (value: number, commit: boolean) => {
      const next = clampFontAxisValue(value, axis);
      // The other axes come from the style at write time, so one changed mid-drag is kept.
      applyEdit((style) => ({ fontVariations: { ...style.fontVariations, [axis.tag]: next } }), commit);
    },
    [applyEdit, axis]
  );
  const previewValue = useCallback((value: number) => setValue(value, false), [setValue]);
  const commitValue = useCallback((value: number) => setValue(value, true), [setValue]);
  const gesture = useSliderGesture(committed, commitValue, previewValue);
  const formatValue = useCallback((value: number) => value.toFixed(precision), [precision]);

  return (
    <ScrubberField
      defaultValue={clampFontAxisValue(axis.default, axis)}
      formatValue={formatValue}
      label={axis.label}
      max={axis.maximum}
      min={axis.minimum}
      step={step}
      value={gesture.value}
      onChange={gesture.onChange}
      onChangeEnd={gesture.onChangeEnd}
    />
  );
};

const variationMapsEqual = (
  left: Readonly<Record<string, number>>,
  right: Readonly<Record<string, number>>,
  axes: readonly FontAxis[]
): boolean => axes.every((axis) => left[axis.tag] === right[axis.tag]);

/** A variable face's named instances and axes, as plain rows of the Font group. */
const FontAxisSettings = ({
  activeVariations,
  applyEdit,
  font,
}: {
  activeVariations: Readonly<Record<string, number>>;
  applyEdit: (edit: TextEdit, commit: boolean) => void;
  font?: FontRecord;
}) => {
  const { t } = useTranslation();
  const [showAdvanced, setShowAdvanced] = useState(false);
  const axes = useMemo(() => usableAxes(font), [font]);
  const visibleAxes = useMemo(
    () => axes.filter((axis) => showAdvanced || !axis.hidden || OWNING_AXIS_TAGS.has(axis.tag)),
    [axes, showAdvanced]
  );
  const hasHiddenAxes = axes.some((axis) => axis.hidden && !OWNING_AXIS_TAGS.has(axis.tag));
  const presetCollection = useMemo(
    () =>
      createListCollection<{ label: string; value: string }>({
        items: [
          { label: t('widgets.canvas.toolOptions.textFontDefaultPreset'), value: 'default' },
          ...(font?.instances ?? []).map((instance, index) => ({ label: instance.name, value: `instance:${index}` })),
        ],
      }),
    [font?.instances, t]
  );
  const presetValue = useMemo(() => {
    if (!font) {
      return ['default'];
    }
    const defaultCoordinates = fontVariationCoordinates(font);
    if (variationMapsEqual(activeVariations, defaultCoordinates, axes)) {
      return ['default'];
    }
    const instanceIndex = font.instances.findIndex((instance) =>
      variationMapsEqual(activeVariations, fontVariationCoordinates(font, instance.coordinates), axes)
    );
    return [instanceIndex >= 0 ? `instance:${instanceIndex}` : 'default'];
  }, [activeVariations, axes, font]);
  const onPreset = useCallback(
    ({ value }: SelectValueChangeDetails<{ label: string; value: string }>) => {
      if (!font) {
        return;
      }
      const selection = value[0];
      if (selection === 'default') {
        applyEdit({ fontVariations: fontVariationCoordinates(font) }, true);
        return;
      }
      const index = selection?.startsWith('instance:') ? Number(selection.slice('instance:'.length)) : -1;
      const instance = Number.isInteger(index) && index >= 0 ? font.instances[index] : undefined;
      if (instance) {
        applyEdit({ fontVariations: fontVariationCoordinates(font, instance.coordinates) }, true);
      }
    },
    [applyEdit, font]
  );
  const resetAxes = useCallback(() => {
    if (font) {
      applyEdit({ fontVariations: fontVariationCoordinates(font) }, true);
    }
  }, [applyEdit, font]);

  if (!font || axes.length === 0) {
    return null;
  }

  const advancedLabel = t(
    showAdvanced
      ? 'widgets.canvas.toolOptions.textFontHideAdvancedAxes'
      : 'widgets.canvas.toolOptions.textFontShowAdvancedAxes'
  );

  return (
    <>
      <PropertyControlRow label={t('widgets.canvas.toolOptions.textFontPreset')}>
        <Select
          aria-label={t('widgets.canvas.toolOptions.textFontPreset')}
          collection={presetCollection}
          minW="0"
          positioning={SELECT_POSITIONING}
          value={presetValue}
          valueText={presetCollection.items.find((item) => item.value === presetValue[0])?.label}
          onValueChange={onPreset}
        />
        <HStack gap="0.5">
          <Tooltip content={t('widgets.canvas.toolOptions.textFontResetAxes')}>
            <IconButton
              aria-label={t('widgets.canvas.toolOptions.textFontResetAxes')}
              size="sm"
              variant="ghost"
              onClick={resetAxes}
            >
              <RotateCcwIcon />
            </IconButton>
          </Tooltip>
          {hasHiddenAxes ? (
            <ToggleIconButton
              checked={showAdvanced}
              icon={SlidersHorizontalIcon}
              label={advancedLabel}
              onCheckedChange={setShowAdvanced}
            />
          ) : null}
        </HStack>
      </PropertyControlRow>
      {visibleAxes.map((axis) => (
        <AxisControl key={axis.tag} activeVariations={activeVariations} applyEdit={applyEdit} axis={axis} />
      ))}
    </>
  );
};

/** One line under the Family row for the catalog's transient states; nothing when it is settled. */
const FontCatalogStatus = ({
  hasNextPage,
  isError,
  isFetchingNextPage,
  isPending,
  onLoadMore,
  onRetry,
}: {
  hasNextPage: boolean;
  isError: boolean;
  isFetchingNextPage: boolean;
  isPending: boolean;
  onLoadMore: () => void;
  onRetry: () => void;
}) => {
  const { t } = useTranslation();
  const { t: tFonts } = useTranslation('fonts');
  if (isPending) {
    return (
      <PropertyControlRow>
        <HStack color="fg.muted" gap="1.5" gridColumn="2 / -1" role="status">
          <Spinner />
          <Text fontSize="xs">{t('common.loading')}</Text>
        </HStack>
      </PropertyControlRow>
    );
  }
  if (isError) {
    return (
      <PropertyControlRow>
        <HStack color="fg.error" gap="1" gridColumn="2 / -1" role="alert">
          <Text fontSize="xs" minW="0" truncate>
            {tFonts('fonts.couldNotLoad')}
          </Text>
          <Button
            aria-label={t('common.retry')}
            disabled={isFetchingNextPage}
            flexShrink="0"
            size="sm"
            variant="ghost"
            onClick={onRetry}
          >
            {t('common.retry')}
          </Button>
        </HStack>
      </PropertyControlRow>
    );
  }
  if (hasNextPage) {
    return (
      <PropertyControlRow>
        <Box gridColumn="2 / -1">
          <Button
            aria-label={tFonts('fonts.loadMore', { defaultValue: 'Load more fonts' })}
            disabled={isFetchingNextPage}
            size="sm"
            variant="ghost"
            onClick={onLoadMore}
          >
            {isFetchingNextPage
              ? tFonts('fonts.loadingMore', { defaultValue: 'Loading more fonts…' })
              : tFonts('fonts.loadMore', { defaultValue: 'Load more fonts' })}
          </Button>
        </Box>
      </PropertyControlRow>
    );
  }
  return null;
};

export const TextFontSettings = ({ engine }: ToolFormProps) => {
  const { t } = useTranslation();
  const { active, applyEdit } = useTextEditor(engine);
  const { data, fetchNextPage, hasNextPage, isError, isFetchNextPageError, isFetchingNextPage, isPending, refetch } =
    useInfiniteQuery(fontsInfiniteQueryOptions({ limit: 100, scope: 'all' }));
  const loadMoreFonts = useCallback(() => void fetchNextPage(), [fetchNextPage]);
  const retryFontCatalog = useCallback(() => {
    if (isFetchNextPageError) {
      void fetchNextPage();
      return;
    }
    void refetch();
  }, [fetchNextPage, isFetchNextPageError, refetch]);
  const fonts = useMemo(() => data?.pages.flatMap((page) => page.items) ?? EMPTY_FONT_RECORDS, [data?.pages]);
  const familyChoices = useMemo(() => {
    const builtins: FontChoice[] = TEXT_FONT_FAMILIES.map((font) => ({
      family: font.value,
      group: 'builtin',
      label: font.label,
      value: `builtin:${font.value}`,
    }));
    const custom = [...fonts]
      .sort((left, right) => left.family.localeCompare(right.family) || left.label.localeCompare(right.label))
      .map((font): FontChoice => ({
        family: font.family,
        font,
        group: 'custom',
        label: font.label,
        ref: fontReferenceFromRecord(font),
        value: fontChoiceValue(font),
      }));
    if (active.fontRef && !custom.some((choice) => sameFontRef(choice.ref, active.fontRef))) {
      custom.push({
        family: active.fontRef.family,
        group: 'custom',
        label: active.fontRef.label,
        ref: { ...active.fontRef },
        value: fontChoiceValue(active.fontRef),
      });
    }
    return [...builtins, ...custom];
  }, [active.fontRef, fonts]);
  const familyCollection = useMemo(() => createListCollection<FontChoice>({ items: familyChoices }), [familyChoices]);
  const groupBy = useCallback((item: FontChoice) => item.group, []);
  const renderGroupLabel = useCallback(
    (group: string) =>
      group === 'custom'
        ? t('widgets.canvas.toolOptions.textFontCustomGroup')
        : t('widgets.canvas.toolOptions.textFontBuiltinGroup'),
    [t]
  );
  const activeChoice = useMemo(
    () =>
      familyChoices.find((choice) =>
        choice.group === 'custom'
          ? sameFontRef(choice.ref, active.fontRef)
          : active.fontRef === undefined && choice.family === active.fontFamily
      ),
    [active.fontFamily, active.fontRef, familyChoices]
  );
  const familyValue = useMemo(
    () => [activeChoice?.value ?? active.fontFamily],
    [active.fontFamily, activeChoice?.value]
  );
  const familyLabel = activeChoice?.label ?? active.fontRef?.label ?? active.fontFamily;
  const activeFont = activeChoice?.font;
  // An axis owns its property: the plain select for it is not offered alongside.
  const activeAxes = useMemo(() => usableAxes(activeFont), [activeFont]);
  const hasWeightAxis = activeAxes.some((axis) => axis.tag === 'wght');
  const hasStyleAxis = activeAxes.some((axis) => axis.tag === 'ital' || axis.tag === 'slnt');
  const onFamily = useCallback(
    ({ value }: SelectValueChangeDetails<FontChoice>) => {
      const choice = familyChoices.find((entry) => entry.value === value[0]);
      if (!choice) {
        return;
      }
      if (choice.group === 'builtin') {
        if (active.fontFamily !== choice.family || active.fontRef !== undefined) {
          applyEdit({ fontFamily: choice.family, fontRef: undefined, fontStyle: 'normal', fontVariations: {} }, true);
        }
        return;
      }
      if (!choice.font || !choice.ref) {
        return;
      }
      if (active.fontFamily === choice.family && sameFontRef(active.fontRef, choice.ref)) {
        return;
      }
      applyEdit(
        {
          fontFamily: choice.family,
          fontRef: choice.ref,
          fontStyle: normalizeFontStyle(choice.font.style),
          fontVariations: fontVariationCoordinates(choice.font),
          fontWeight:
            Number.isFinite(choice.font.weight) && choice.font.weight > 0 ? choice.font.weight : active.fontWeight,
        },
        true
      );
    },
    [active.fontFamily, active.fontRef, active.fontWeight, applyEdit, familyChoices]
  );
  const styleCollection = useMemo(
    () =>
      createListCollection<{ label: string; value: NonNullable<TextSource['fontStyle']> }>({
        items: [
          { label: t('widgets.canvas.toolOptions.textFontStyleNormal'), value: 'normal' },
          { label: t('widgets.canvas.toolOptions.textFontStyleItalic'), value: 'italic' },
          { label: t('widgets.canvas.toolOptions.textFontStyleOblique'), value: 'oblique' },
        ],
      }),
    [t]
  );
  const onStyle = useCallback(
    ({ value }: SelectValueChangeDetails<{ label: string; value: NonNullable<TextSource['fontStyle']> }>) => {
      const next = value[0];
      if ((next === 'normal' || next === 'italic' || next === 'oblique') && next !== active.fontStyle) {
        applyEdit({ fontStyle: next }, true);
      }
    },
    [active.fontStyle, applyEdit]
  );
  const weightValues = useMemo(
    () =>
      [...new Set([...TEXT_FONT_WEIGHTS, active.fontWeight].filter((weight) => Number.isFinite(weight)))].sort(
        (a, b) => a - b
      ),
    [active.fontWeight]
  );
  const weightCollection = useMemo(
    () =>
      createListCollection<{ label: string; value: string }>({
        items: weightValues.map((weight) => ({ label: String(weight), value: String(weight) })),
      }),
    [weightValues]
  );
  const weightValue = useMemo(() => [String(active.fontWeight)], [active.fontWeight]);
  const styleValue = useMemo(() => [active.fontStyle], [active.fontStyle]);
  const onWeight = useCallback(
    ({ value }: SelectValueChangeDetails<{ label: string; value: string }>) => {
      const next = value[0] ? Number(value[0]) : undefined;
      if (next !== undefined && Number.isFinite(next) && next !== active.fontWeight) {
        applyEdit({ fontWeight: next }, true);
      }
    },
    [active.fontWeight, applyEdit]
  );
  // Ticks preview through the defaults/session; ONE document commit lands on release.
  const previewSize = useCallback((value: number) => applyEdit({ fontSize: clampTextSize(value) }, false), [applyEdit]);
  const setSize = useCallback((value: number) => applyEdit({ fontSize: clampTextSize(value) }, true), [applyEdit]);
  const sizeGesture = useSliderGesture(Math.round(active.fontSize), setSize, previewSize);
  const previewLineHeight = useCallback(
    (value: number) => applyEdit({ lineHeight: Math.max(0.5, Math.round(value * 10) / 10) }, false),
    [applyEdit]
  );
  const setLineHeight = useCallback(
    (value: number) => applyEdit({ lineHeight: Math.max(0.5, Math.round(value * 10) / 10) }, true),
    [applyEdit]
  );
  const lineHeightGesture = useSliderGesture(active.lineHeight, setLineHeight, previewLineHeight);
  const showStyle = !hasStyleAxis;
  const showWeight = !hasWeightAxis;
  return (
    <>
      <PropertyControlRow label={t('widgets.properties.rows.family')}>
        <Select
          aria-label={t('widgets.canvas.toolOptions.textFont')}
          collection={familyCollection}
          gridColumn="2 / -1"
          groupBy={groupBy}
          itemsMaxH="20rem"
          minW="0"
          positioning={SELECT_POSITIONING}
          renderGroupLabel={renderGroupLabel}
          value={familyValue}
          valueText={familyLabel}
          onValueChange={onFamily}
        />
      </PropertyControlRow>
      <FontCatalogStatus
        hasNextPage={hasNextPage}
        isError={isError}
        isFetchingNextPage={isFetchingNextPage}
        isPending={isPending}
        onLoadMore={loadMoreFonts}
        onRetry={retryFontCatalog}
      />
      {showStyle || showWeight ? (
        <PropertyControlRow
          label={t(showStyle ? 'widgets.canvas.toolOptions.textFontStyle' : 'widgets.properties.rows.weight')}
        >
          {showStyle ? (
            <Select
              aria-label={t('widgets.canvas.toolOptions.textFontStyle')}
              collection={styleCollection}
              gridColumn={showWeight ? undefined : '2 / -1'}
              minW="0"
              positioning={SELECT_POSITIONING}
              value={styleValue}
              valueText={styleCollection.items.find((item) => item.value === active.fontStyle)?.label}
              onValueChange={onStyle}
            />
          ) : null}
          {showWeight ? (
            <Select
              aria-label={t('widgets.canvas.toolOptions.textWeight')}
              collection={weightCollection}
              flexShrink={0}
              positioning={SELECT_POSITIONING}
              triggerProps={WEIGHT_TRIGGER_PROPS}
              value={weightValue}
              valueText={String(active.fontWeight)}
              w="4.5rem"
              onValueChange={onWeight}
            />
          ) : null}
        </PropertyControlRow>
      ) : null}
      <ScrubberField
        defaultValue={DEFAULT_TEXT_OPTIONS.fontSize}
        formatValue={formatTextSize}
        inputMax={MAX_TEXT_FONT_SIZE}
        inputMin={MIN_TEXT_FONT_SIZE}
        label={t('widgets.canvas.toolOptions.textSize')}
        max={SIZE_TRACK_MAX_PX}
        min={SIZE_TRACK_MIN_PX}
        scale="log"
        step={1}
        stepFor={textSizeKeyboardStep}
        value={sizeGesture.value}
        onChange={sizeGesture.onChange}
        onChangeEnd={sizeGesture.onChangeEnd}
      />
      <ScrubberField
        defaultValue={DEFAULT_TEXT_OPTIONS.lineHeight}
        formatValue={formatLineHeight}
        label={t('widgets.canvas.toolOptions.textLineHeight')}
        max={4}
        min={0.5}
        step={0.1}
        value={lineHeightGesture.value}
        onChange={lineHeightGesture.onChange}
        onChangeEnd={lineHeightGesture.onChangeEnd}
      />
      <FontAxisSettings activeVariations={active.fontVariations} applyEdit={applyEdit} font={activeFont} />
    </>
  );
};

export const TextParagraphSettings = ({ engine }: ToolFormProps) => {
  const { t } = useTranslation();
  const { active, applyEdit } = useTextEditor(engine);
  const onAlign = useCallback((next: TextAlign) => applyEdit({ align: next }, true), [applyEdit]);
  return (
    <PropertyControlRow label={t('widgets.properties.rows.align')}>
      <HStack aria-label={t('widgets.properties.rows.align')} gap="0.5" role="group">
        {ALIGN_VALUES.map((value) => (
          <AlignButton key={value} active={active.align === value} value={value} onSelect={onAlign} />
        ))}
      </HStack>
    </PropertyControlRow>
  );
};

export const TextColorSettings = ({ engine }: ToolFormProps) => {
  const { t } = useTranslation();
  const { active, applyEdit } = useTextEditor(engine);
  const sampleColor = useColorSampler(engine);
  const onChange = useCallback((hex: string) => applyEdit({ color: hex }, false), [applyEdit]);
  const onChangeEnd = useCallback((hex: string) => applyEdit({ color: hex }, true), [applyEdit]);
  return (
    <PropertyControlRow label={t('widgets.properties.rows.fill')}>
      <ColorPicker
        aria-label={t('widgets.canvas.toolOptions.textColor')}
        value={active.color}
        onSampleColor={sampleColor}
        onValueChange={onChange}
        onValueChangeEnd={onChangeEnd}
      />
    </PropertyControlRow>
  );
};
