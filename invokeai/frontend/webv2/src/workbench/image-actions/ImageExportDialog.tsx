import type {
  ImageExportEncoding,
  ImageExportFormat,
  ImageExportMetadata,
  ImageExportSettings,
  ImageSize,
} from '@features/gallery/contracts';
import type { KeyboardEvent, ReactNode, SyntheticEvent } from 'react';

import {
  Box,
  Flex,
  Grid,
  HStack,
  Image,
  Input,
  NumberInput,
  Portal,
  Stack,
  Text,
  VisuallyHidden,
} from '@chakra-ui/react';
import { GALLERY_REQUEST_MAX_NAMES } from '@features/gallery';
import {
  getImageExportDimensions,
  getImageExportEncoding,
  getImageExportLongEdgeForScale,
  getImageExportLongEdgeForSide,
  IMAGE_EXPORT_FORMAT_LABELS,
  IMAGE_EXPORT_FORMATS,
  IMAGE_EXPORT_NAME_PATTERN_MAX_LENGTH,
  IMAGE_EXPORT_NAME_TOKENS,
  isLossyImageExportFormat,
} from '@features/gallery/contracts';
import {
  galleryImageExportNamesOptions,
  galleryImageExportOptions,
  galleryImageFileSizeOptions,
} from '@features/gallery/queries';
import { galleryImageUrls } from '@features/gallery/utility';
import { formatBytes, formatCount } from '@platform/i18n/languages';
import { useDebouncedValue } from '@platform/react/useDebouncedValue';
import { useMountEffect } from '@platform/react/useMountEffect';
import { isAccountScopeCurrent } from '@platform/state/accountLifecycle';
import { ApiError, getApiErrorMessage } from '@platform/transport/http';
import { Button, CloseButton } from '@platform/ui/Button';
import { Dialog } from '@platform/ui/Dialog';
import { Field, FieldLabel } from '@platform/ui/Field';
import { MiddleTruncate } from '@platform/ui/MiddleTruncate';
import { SegmentedControl } from '@platform/ui/SegmentedControl';
import { SliderNumberField } from '@platform/ui/SliderNumberField';
import { keepPreviousData, useQuery } from '@tanstack/react-query';
import { patchWorkbenchPreferences } from '@workbench/settings/store';
import { ImageUpIcon } from 'lucide-react';
import { useCallback, useId, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

import type { ImageExportDialogRequest } from './useImageExportDialog';

import { formatImageExportEncoding, getImageExportTargetNames, useImageExportPreference } from './imageExport';
import { ImageExportCompare, type ImageExportPreviewError } from './ImageExportCompare';

const PREVIEW_DEBOUNCE_MS = 350;
const NAME_DEBOUNCE_MS = 300;
const PREVIEW_STRIP_LIMIT = 12;
const DEFAULT_SELECTION_LONG_EDGE = 2048;
const SCALE_SHORTCUTS = [1, 0.5, 0.25] as const;
const NAME_SEPARATORS = new Set(['-', '_', '.', ' ']);
const EXAMPLE_TAIL_GRAPHEMES = 12;

const BODY_COLUMNS = { base: 'minmax(0, 1fr)', md: 'minmax(0, 1fr) 18rem' } as const;
const BODY_HEIGHT = { base: 'auto', md: 'min(38rem, calc(100dvh - 9rem))' } as const;
const BODY_ROWS = { base: 'auto', md: 'minmax(0, 1fr)' } as const;
const BODY_OVERFLOW = { base: 'auto', md: 'hidden' } as const;
const PREVIEW_BORDER_BOTTOM = { base: '1px', md: '0' } as const;
const PREVIEW_BORDER_END = { base: '0', md: '1px' } as const;
const PREVIEW_HEIGHT = { base: '22rem', md: 'auto' } as const;
const CONTROLS_OVERFLOW = { base: 'visible', md: 'auto' } as const;
const TILE_FOCUS_RING = { outline: '2px solid', outlineColor: 'accent.focusRing', outlineOffset: '1px' } as const;

const isSameEncoding = (left: ImageExportEncoding, right: ImageExportEncoding): boolean =>
  left.format === right.format &&
  left.longEdge === right.longEdge &&
  left.metadata === right.metadata &&
  left.quality === right.quality;

const isPermanentRefusal = (error: unknown): boolean =>
  error instanceof ApiError &&
  error.status >= 400 &&
  error.status < 500 &&
  error.status !== 408 &&
  error.status !== 409 &&
  error.status !== 429;

export const ImageExportDialog = ({
  isOpen,
  onClose,
  onExitComplete,
  request,
}: {
  isOpen: boolean;
  onClose: () => void;
  onExitComplete: () => void;
  request: ImageExportDialogRequest;
}) => {
  const { i18n, t } = useTranslation();
  const { owner, target } = request;
  const savedSettings = useImageExportPreference();
  const [settings, setSettings] = useState(savedSettings);
  const [previewIndex, setPreviewIndex] = useState(0);
  const [loadedOriginal, setLoadedOriginal] = useState<{ name: string; size: ImageSize } | null>(null);
  const [loadedExport, setLoadedExport] = useState<{ blob: Blob; size: ImageSize } | null>(null);
  const [isExporting, setIsExporting] = useState(false);
  const [exportError, setExportError] = useState<string | null>(null);
  const exportAbortRef = useRef<AbortController | null>(null);
  const imageNames = getImageExportTargetNames(target);
  const isSelection = target.kind === 'selection';
  const image = target.kind === 'image' ? target.image : null;
  const previewName = imageNames[previewIndex] ?? '';
  const sourceSize: ImageSize | null = image ?? (loadedOriginal?.name === previewName ? loadedOriginal.size : null);
  const patchSettings = useCallback((changes: Partial<ImageExportSettings>) => {
    setSettings((current) => ({ ...current, ...changes }));
    setExportError(null);
  }, []);

  useMountEffect(() => () => exportAbortRef.current?.abort());

  const encoding = getImageExportEncoding(settings, image ?? undefined);
  const previewEncoding = useDebouncedValue(encoding, PREVIEW_DEBOUNCE_MS, { isEqual: isSameEncoding });
  const exportQuery = useQuery({
    ...galleryImageExportOptions(owner, previewName, previewEncoding),
    enabled: isOpen && previewName !== '',
    // Another image's encode never stands in for this one's.
    placeholderData: (previous, previousQuery) => (previousQuery?.queryKey[2] === previewName ? previous : undefined),
  });
  const originalSizeQuery = useQuery({
    ...galleryImageFileSizeOptions(owner, previewName),
    enabled: isOpen && previewName !== '',
  });
  const namedPattern = useDebouncedValue(settings.namePattern, NAME_DEBOUNCE_MS);
  const archiveStart = previewIndex - (previewIndex % GALLERY_REQUEST_MAX_NAMES);
  const namesQuery = useQuery({
    ...galleryImageExportNamesOptions(owner, {
      format: settings.format,
      imageNames: imageNames.slice(archiveStart, previewIndex + 1),
      pattern: namedPattern,
    }),
    enabled: isOpen && previewName !== '' && namedPattern.trim() !== '',
    placeholderData: keepPreviousData,
  });

  const exportBlob = exportQuery.data ?? null;
  const isEncodingSettled = isSameEncoding(encoding, previewEncoding);
  const isEncoding = exportQuery.isFetching || !isEncodingSettled;
  const exportSize = exportBlob && loadedExport?.blob === exportBlob ? loadedExport.size : null;
  const patternIsBlank = settings.namePattern.trim() === '';
  const lossy = isLossyImageExportFormat(settings.format);
  const hasPreviewError = exportQuery.isError && !exportQuery.isPlaceholderData;
  const isPreviewRefused = hasPreviewError && isPermanentRefusal(exportQuery.error);
  const previewError = useMemo<ImageExportPreviewError | null>(() => {
    if (!hasPreviewError) {
      return null;
    }

    const detail = getApiErrorMessage(exportQuery.error, '');

    return isPreviewRefused
      ? { canRetry: false, detail, message: t('widgets.gallery.export.previewRefused') }
      : { canRetry: true, detail, message: t('widgets.gallery.export.previewFailed') };
  }, [exportQuery.error, hasPreviewError, isPreviewRefused, t]);
  const refusalReason =
    isPreviewRefused && isEncodingSettled ? previewError?.detail || t('widgets.gallery.export.previewRefused') : null;

  const handleOriginalLoad = useCallback(
    (size: ImageSize) => setLoadedOriginal({ name: previewName, size }),
    [previewName]
  );
  const handleExportLoad = useCallback(
    (size: ImageSize) => {
      if (exportBlob) {
        setLoadedExport({ blob: exportBlob, size });
      }
    },
    [exportBlob]
  );
  const retryPreview = useCallback(() => exportQuery.refetch(), [exportQuery]);

  const handleOpenChange = useCallback(
    ({ open }: { open: boolean }) => {
      if (!open) {
        exportAbortRef.current?.abort();
        onClose();
      }
    },
    [onClose]
  );
  const close = useCallback(() => handleOpenChange({ open: false }), [handleOpenChange]);

  const handleExport = useCallback(async () => {
    if (isExporting || patternIsBlank || imageNames.length === 0 || refusalReason !== null) {
      return;
    }

    if (target.kind === 'selection') {
      void patchWorkbenchPreferences({ imageExportSettings: { ...settings } });
      close();
      void request.execute(target, settings);
      return;
    }

    const controller = new AbortController();

    exportAbortRef.current = controller;
    setIsExporting(true);
    setExportError(null);
    try {
      await request.execute(target, settings, { signal: controller.signal });
      void patchWorkbenchPreferences({ imageExportSettings: { ...settings } });
      // Closed or replaced meanwhile, the host may already show another request.
      if (!controller.signal.aborted) {
        close();
      }
    } catch (error: unknown) {
      if (!controller.signal.aborted && isAccountScopeCurrent(owner)) {
        setExportError(getApiErrorMessage(error, t('widgets.gallery.export.exportFailed')));
      }
    } finally {
      if (exportAbortRef.current === controller) {
        exportAbortRef.current = null;
      }
      setIsExporting(false);
    }
  }, [close, imageNames.length, isExporting, owner, patternIsBlank, refusalReason, request, settings, t, target]);

  const subtitle =
    target.kind === 'image'
      ? originalSizeQuery.data
        ? t('widgets.gallery.export.subtitleImageWithSize', {
            height: formatCount(target.image.height),
            name: target.image.imageName,
            size: formatBytes(originalSizeQuery.data),
            width: formatCount(target.image.width),
          })
        : t('widgets.gallery.export.subtitleImage', {
            height: formatCount(target.image.height),
            name: target.image.imageName,
            width: formatCount(target.image.width),
          })
      : target.videoCount > 0
        ? t('widgets.gallery.export.videosLeftOut', { count: target.videoCount })
        : null;

  return (
    <Dialog.Root
      closeOnInteractOutside={!isExporting}
      finalFocusEl={request.returnFocus}
      open={isOpen}
      placement="center"
      size="xl"
      onExitComplete={onExitComplete}
      onOpenChange={handleOpenChange}
    >
      <Portal>
        <Dialog.Backdrop />
        <Dialog.Positioner>
          <Dialog.Content maxW="60rem" overflow="hidden" p="0" w="calc(100dvw - 2rem)">
            <Dialog.Header alignItems="baseline" borderBottomWidth="1px" borderColor="border.subtle" gap="2" pe="10">
              <Dialog.Title flexShrink={0}>
                {isSelection
                  ? t('widgets.gallery.export.titleSelection', { count: imageNames.length })
                  : t('widgets.gallery.export.titleImage')}
              </Dialog.Title>
              {subtitle ? (
                <Dialog.Description
                  fontSize="sm"
                  minW="0"
                  overflow="hidden"
                  textOverflow="ellipsis"
                  whiteSpace="nowrap"
                >
                  {subtitle}
                </Dialog.Description>
              ) : null}
            </Dialog.Header>
            <Grid
              gridTemplateColumns={BODY_COLUMNS}
              gridTemplateRows={BODY_ROWS}
              h={BODY_HEIGHT}
              maxH="calc(100dvh - 9rem)"
              overflowY={BODY_OVERFLOW}
            >
              <Stack
                borderColor="border.subtle"
                borderBottomWidth={PREVIEW_BORDER_BOTTOM}
                borderEndWidth={PREVIEW_BORDER_END}
                gap="0"
                h={PREVIEW_HEIGHT}
                minH="0"
                minW="0"
              >
                {previewName === '' ? (
                  <Flex align="center" bg="bg.inset" color="fg.muted" flex="1" fontSize="md" justify="center" p="6">
                    {t('widgets.gallery.export.noImagesInSelection')}
                  </Flex>
                ) : (
                  <ImageExportCompare
                    error={previewError}
                    exportBlob={exportBlob}
                    exportLabel={t('widgets.gallery.export.exportSide', {
                      encoding: formatImageExportEncoding(t, settings),
                    })}
                    isEncoding={isEncoding}
                    isRetrying={exportQuery.isFetching}
                    originalAlt={t('widgets.gallery.export.original')}
                    originalUrl={image ? image.imageUrl : galleryImageUrls.full(previewName)}
                    sourceSize={sourceSize}
                    onExportLoad={handleExportLoad}
                    onOriginalLoad={handleOriginalLoad}
                    onRetry={retryPreview}
                  />
                )}
                {isSelection && imageNames.length > 1 ? (
                  <PreviewStrip imageNames={imageNames} selectedIndex={previewIndex} onSelect={setPreviewIndex} />
                ) : null}
              </Stack>
              <Box minH="0" minW="0" overflowY={CONTROLS_OVERFLOW}>
                <Stack gap="4" px="4" py="3">
                  <FormatControls disabled={isExporting} lossy={lossy} settings={settings} onChange={patchSettings} />
                  {image ? (
                    <ImageSizeControls
                      disabled={isExporting}
                      settings={settings}
                      source={image}
                      onChange={patchSettings}
                    />
                  ) : (
                    <SelectionSizeControls disabled={isExporting} settings={settings} onChange={patchSettings} />
                  )}
                  <MetadataControls disabled={isExporting} settings={settings} onChange={patchSettings} />
                  <FileNameControls
                    disabled={isExporting}
                    exampleName={namesQuery.data?.at(-1) ?? null}
                    isExampleError={namesQuery.isError}
                    isExamplePending={namesQuery.isFetching || namedPattern !== settings.namePattern}
                    settings={settings}
                    onChange={patchSettings}
                    onSubmit={handleExport}
                  />
                </Stack>
              </Box>
            </Grid>
            <Dialog.Footer
              bg="bg.muted"
              borderColor="border.subtle"
              borderTopWidth="1px"
              flexWrap="wrap"
              justifyContent="flex-end"
              pt="2.5"
            >
              <Box flex="1" minW="12rem">
                {exportError ? (
                  <Text color="fg.error" fontSize="sm" role="alert">
                    {exportError}
                  </Text>
                ) : refusalReason ? (
                  <Text color="fg.error" fontSize="sm">
                    {refusalReason}
                  </Text>
                ) : (
                  <ExportSummary
                    exportBytes={exportBlob?.size ?? null}
                    exportSize={exportSize}
                    isEncoding={isEncoding}
                    isSelection={isSelection}
                    language={i18n.language}
                    originalBytes={originalSizeQuery.data ?? null}
                  />
                )}
              </Box>
              <Button color="fg" variant="ghost" onClick={close}>
                {t('common.cancel')}
              </Button>
              <Button
                colorPalette="accent"
                disabled={patternIsBlank || imageNames.length === 0}
                loading={isExporting}
                loadingText={t('widgets.gallery.export.exporting')}
                unavailableReason={refusalReason ?? undefined}
                onClick={handleExport}
              >
                <ImageUpIcon />
                {isSelection
                  ? t('widgets.gallery.export.exportCount', { count: imageNames.length })
                  : t('widgets.gallery.export.export')}
              </Button>
            </Dialog.Footer>
            <Dialog.CloseTrigger asChild>
              <CloseButton aria-label={t('common.close')} size="sm" />
            </Dialog.CloseTrigger>
          </Dialog.Content>
        </Dialog.Positioner>
      </Portal>
    </Dialog.Root>
  );
};

type ChangeSettings = (changes: Partial<ImageExportSettings>) => void;

/** Not a Field: it would hand its one control id to inputs that bring their own (segments, sliders). */
const ControlGroup = ({ children, hint, label }: { children: ReactNode; hint?: string; label: string }) => {
  const labelId = useId();
  const hintId = useId();

  return (
    <Stack aria-describedby={hint ? hintId : undefined} aria-labelledby={labelId} gap="1.5" minW="0" role="group">
      <Box id={labelId}>
        <FieldLabel>{label}</FieldLabel>
      </Box>
      {children}
      {hint ? (
        <Text color="fg.muted" fontSize="xs" id={hintId}>
          {hint}
        </Text>
      ) : null}
    </Stack>
  );
};

const FormatControls = ({
  disabled,
  lossy,
  onChange,
  settings,
}: {
  disabled: boolean;
  lossy: boolean;
  onChange: ChangeSettings;
  settings: ImageExportSettings;
}) => {
  const { t } = useTranslation();
  const formatOptions = useMemo(
    () => IMAGE_EXPORT_FORMATS.map((format) => ({ label: IMAGE_EXPORT_FORMAT_LABELS[format], value: format })),
    []
  );
  const handleFormatChange = useCallback(
    (value: string) => onChange({ format: value as ImageExportFormat }),
    [onChange]
  );
  const handleQualityChange = useCallback((quality: number) => onChange({ quality }), [onChange]);

  return (
    <>
      <ControlGroup
        hint={t(`widgets.gallery.export.formatHint.${settings.format}`)}
        label={t('widgets.gallery.export.format')}
      >
        <SegmentedControl
          ariaLabel={t('widgets.gallery.export.format')}
          disabled={disabled}
          options={formatOptions}
          value={settings.format}
          onChange={handleFormatChange}
        />
      </ControlGroup>
      <ControlGroup
        hint={lossy ? undefined : t('widgets.gallery.export.qualityLossless')}
        label={t('widgets.gallery.export.quality')}
      >
        <SliderNumberField
          ariaLabel={t('widgets.gallery.export.quality')}
          disabled={disabled || !lossy}
          max={100}
          min={1}
          step={1}
          value={settings.quality}
          onChange={handleQualityChange}
        />
      </ControlGroup>
    </>
  );
};

const CommittedNumberInput = ({
  ariaLabel,
  disabled,
  max,
  value,
  onCommit,
}: {
  ariaLabel: string;
  disabled: boolean;
  max?: number;
  value: number;
  onCommit: (value: number) => void;
}) => {
  const [draft, setDraft] = useState<string | null>(null);
  const handleValueChange = useCallback(({ value: next }: NumberInput.ValueChangeDetails) => setDraft(next), []);
  const handleValueCommit = useCallback(
    ({ valueAsNumber }: NumberInput.ValueChangeDetails) => {
      setDraft(null);
      if (Number.isFinite(valueAsNumber) && valueAsNumber >= 1) {
        onCommit(Math.round(valueAsNumber));
      }
    },
    [onCommit]
  );

  return (
    <NumberInput.Root
      disabled={disabled}
      flex="1"
      max={max}
      min={1}
      minW="0"
      step={1}
      value={draft ?? String(value)}
      onValueChange={handleValueChange}
      onValueCommit={handleValueCommit}
    >
      <NumberInput.Input aria-label={ariaLabel} fontVariantNumeric="tabular-nums" />
    </NumberInput.Root>
  );
};

const ImageSizeControls = ({
  disabled,
  onChange,
  settings,
  source,
}: {
  disabled: boolean;
  onChange: ChangeSettings;
  settings: ImageExportSettings;
  source: ImageSize;
}) => {
  const { t } = useTranslation();
  const output = getImageExportDimensions(source, settings.longEdge);
  const scaleOptions = useMemo(
    () =>
      SCALE_SHORTCUTS.map((scale) => ({
        label: t('widgets.gallery.export.scalePercent', { percent: scale * 100 }),
        value: String(scale),
      })),
    [t]
  );
  const effectiveLongEdge = getImageExportEncoding(settings, source).longEdge;
  const scaleValue =
    SCALE_SHORTCUTS.find((scale) => getImageExportLongEdgeForScale(source, scale) === effectiveLongEdge) ?? null;
  const handleScaleChange = useCallback(
    (value: string) => onChange({ longEdge: getImageExportLongEdgeForScale(source, Number(value)) }),
    [onChange, source]
  );
  const handleWidthCommit = useCallback(
    (width: number) => onChange({ longEdge: getImageExportLongEdgeForSide(source, 'width', width) }),
    [onChange, source]
  );
  const handleHeightCommit = useCallback(
    (height: number) => onChange({ longEdge: getImageExportLongEdgeForSide(source, 'height', height) }),
    [onChange, source]
  );

  return (
    <ControlGroup hint={t('widgets.gallery.export.sizeHint')} label={t('widgets.gallery.export.size')}>
      <Stack gap="2" w="full">
        <HStack gap="1.5">
          <CommittedNumberInput
            ariaLabel={t('widgets.gallery.export.width')}
            disabled={disabled}
            max={source.width}
            value={output.width}
            onCommit={handleWidthCommit}
          />
          <Text aria-hidden color="fg.subtle" fontSize="sm">
            ×
          </Text>
          <CommittedNumberInput
            ariaLabel={t('widgets.gallery.export.height')}
            disabled={disabled}
            max={source.height}
            value={output.height}
            onCommit={handleHeightCommit}
          />
          <Text color="fg.subtle" fontSize="sm">
            {t('widgets.gallery.export.pixels')}
          </Text>
        </HStack>
        <SegmentedControl
          ariaLabel={t('widgets.gallery.export.scale')}
          disabled={disabled}
          options={scaleOptions}
          value={scaleValue === null ? null : String(scaleValue)}
          onChange={handleScaleChange}
        />
      </Stack>
    </ControlGroup>
  );
};

const SelectionSizeControls = ({
  disabled,
  onChange,
  settings,
}: {
  disabled: boolean;
  onChange: ChangeSettings;
  settings: ImageExportSettings;
}) => {
  const { t } = useTranslation();
  const [limit, setLimit] = useState(settings.longEdge ?? DEFAULT_SELECTION_LONG_EDGE);
  const modeOptions = useMemo(
    () => [
      { label: t('widgets.gallery.export.originalSize'), value: 'original' },
      { label: t('widgets.gallery.export.limitLongEdge'), value: 'limit' },
    ],
    [t]
  );
  const handleModeChange = useCallback(
    (value: string) => onChange({ longEdge: value === 'limit' ? limit : null }),
    [limit, onChange]
  );
  const handleLimitCommit = useCallback(
    (value: number) => {
      setLimit(value);
      onChange({ longEdge: value });
    },
    [onChange]
  );

  return (
    <ControlGroup hint={t('widgets.gallery.export.sizeHint')} label={t('widgets.gallery.export.size')}>
      <Stack gap="2" w="full">
        <SegmentedControl
          ariaLabel={t('widgets.gallery.export.size')}
          disabled={disabled}
          options={modeOptions}
          value={settings.longEdge === null ? 'original' : 'limit'}
          onChange={handleModeChange}
        />
        {settings.longEdge === null ? null : (
          <HStack gap="1.5">
            <CommittedNumberInput
              ariaLabel={t('widgets.gallery.export.longEdge')}
              disabled={disabled}
              value={settings.longEdge}
              onCommit={handleLimitCommit}
            />
            <Text color="fg.subtle" fontSize="sm" whiteSpace="nowrap">
              {t('widgets.gallery.export.longEdgeUnit')}
            </Text>
          </HStack>
        )}
      </Stack>
    </ControlGroup>
  );
};

const METADATA_OPTIONS: readonly { labelKey: string; value: ImageExportMetadata }[] = [
  { labelKey: 'widgets.gallery.export.metadataAll', value: 'all' },
  { labelKey: 'widgets.gallery.export.metadataParameters', value: 'parameters' },
  { labelKey: 'widgets.gallery.export.metadataNone', value: 'none' },
];

const MetadataControls = ({
  disabled,
  onChange,
  settings,
}: {
  disabled: boolean;
  onChange: ChangeSettings;
  settings: ImageExportSettings;
}) => {
  const { t } = useTranslation();
  const isPng = settings.format === 'png';
  const options = useMemo(() => METADATA_OPTIONS.map(({ labelKey, value }) => ({ label: t(labelKey), value })), [t]);
  const handleChange = useCallback((value: string) => onChange({ metadata: value as ImageExportMetadata }), [onChange]);

  return (
    <ControlGroup
      hint={
        isPng
          ? t(`widgets.gallery.export.metadataHint.${settings.metadata}`)
          : t('widgets.gallery.export.metadataPngOnly')
      }
      label={t('widgets.gallery.export.metadata')}
    >
      <SegmentedControl
        ariaLabel={t('widgets.gallery.export.metadata')}
        disabled={disabled || !isPng}
        options={options}
        value={isPng ? settings.metadata : 'none'}
        onChange={handleChange}
      />
    </ControlGroup>
  );
};

const insertNameToken = (pattern: string, at: number, token: string): { caret: number; pattern: string } => {
  const before = pattern.slice(0, at);
  const insertion = before !== '' && !NAME_SEPARATORS.has(before.at(-1) ?? '') ? `-${token}` : token;

  return { caret: at + insertion.length, pattern: `${before}${insertion}${pattern.slice(at)}` };
};

const FileNameControls = ({
  disabled,
  exampleName,
  isExampleError,
  isExamplePending,
  onChange,
  onSubmit,
  settings,
}: {
  disabled: boolean;
  exampleName: string | null;
  isExampleError: boolean;
  isExamplePending: boolean;
  onChange: ChangeSettings;
  onSubmit: () => void;
  settings: ImageExportSettings;
}) => {
  const { t } = useTranslation();
  const inputId = useId();
  const exampleId = useId();
  const caretRef = useRef<number | null>(null);
  const isBlank = settings.namePattern.trim() === '';
  const rememberCaret = useCallback((event: SyntheticEvent<HTMLInputElement>) => {
    caretRef.current = event.currentTarget.selectionEnd;
  }, []);
  const handleChange = useCallback(
    (event: SyntheticEvent<HTMLInputElement>) => {
      caretRef.current = event.currentTarget.selectionEnd;
      onChange({ namePattern: event.currentTarget.value });
    },
    [onChange]
  );
  const handleKeyDown = useCallback(
    (event: KeyboardEvent<HTMLInputElement>) => {
      if (event.key === 'Enter' && !event.nativeEvent.isComposing) {
        event.preventDefault();
        onSubmit();
      }
    },
    [onSubmit]
  );
  const insertToken = useCallback(
    (token: string) => {
      const pattern = settings.namePattern;
      const next = insertNameToken(pattern, Math.min(caretRef.current ?? pattern.length, pattern.length), token);

      if (next.pattern.length <= IMAGE_EXPORT_NAME_PATTERN_MAX_LENGTH) {
        caretRef.current = next.caret;
        onChange({ namePattern: next.pattern });
      }
    },
    [onChange, settings.namePattern]
  );
  const exampleMessage = isBlank
    ? t('widgets.gallery.export.fileNameRequired')
    : isExampleError
      ? t('widgets.gallery.export.fileNameExampleFailed')
      : exampleName
        ? null
        : t('widgets.gallery.export.fileNameResolving');

  return (
    <Field id={inputId} invalid={isBlank} label={t('widgets.gallery.export.fileName')}>
      <Stack gap="1.5" w="full">
        <Input
          aria-describedby={exampleId}
          autoComplete="off"
          disabled={disabled}
          fontFamily="mono"
          maxLength={IMAGE_EXPORT_NAME_PATTERN_MAX_LENGTH}
          spellCheck={false}
          value={settings.namePattern}
          onChange={handleChange}
          onKeyDown={handleKeyDown}
          onSelect={rememberCaret}
        />
        <HStack
          aria-live="polite"
          color={isBlank || isExampleError ? 'fg.error' : 'fg.muted'}
          fontFamily="mono"
          fontSize="xs"
          gap="1"
          id={exampleId}
          minW="0"
          opacity={isExamplePending && !isBlank ? 0.6 : 1}
        >
          {exampleMessage === null && exampleName ? (
            <>
              <Text aria-hidden as="span" flexShrink={0}>
                →
              </Text>
              <MiddleTruncate tailGraphemes={EXAMPLE_TAIL_GRAPHEMES} text={exampleName} />
            </>
          ) : (
            <Text as="span">{exampleMessage}</Text>
          )}
        </HStack>
        <Flex aria-label={t('widgets.gallery.export.tokens')} gap="1" role="group" wrap="wrap">
          {IMAGE_EXPORT_NAME_TOKENS.map((token) => (
            <TokenButton key={token} disabled={disabled} token={token} onInsert={insertToken} />
          ))}
        </Flex>
      </Stack>
    </Field>
  );
};

const TokenButton = ({
  disabled,
  onInsert,
  token,
}: {
  disabled: boolean;
  onInsert: (token: string) => void;
  token: string;
}) => {
  const { t } = useTranslation();
  const handleClick = useCallback(() => onInsert(token), [onInsert, token]);

  return (
    <Button
      aria-label={t('widgets.gallery.export.insertToken', { token })}
      disabled={disabled}
      fontFamily="mono"
      size="xs"
      variant="outline"
      onClick={handleClick}
    >
      {token}
    </Button>
  );
};

const PreviewStrip = ({
  imageNames,
  onSelect,
  selectedIndex,
}: {
  imageNames: readonly string[];
  onSelect: (index: number) => void;
  selectedIndex: number;
}) => {
  const { t } = useTranslation();
  const shown = imageNames.slice(0, PREVIEW_STRIP_LIMIT);
  const hiddenCount = imageNames.length - shown.length;

  return (
    <HStack
      aria-label={t('widgets.gallery.export.previewStrip')}
      bg="bg.inset"
      gap="1"
      overflowX="auto"
      px="3"
      py="2"
      role="group"
    >
      {shown.map((name, index) => (
        <PreviewStripTile
          key={name}
          index={index}
          isSelected={index === selectedIndex}
          name={name}
          onSelect={onSelect}
        />
      ))}
      {hiddenCount > 0 ? (
        <Text color="fg.muted" flexShrink={0} fontSize="sm" fontVariantNumeric="tabular-nums" px="1">
          <span aria-hidden>{t('widgets.gallery.export.moreImages', { count: hiddenCount })}</span>
          <VisuallyHidden>{t('widgets.gallery.export.moreImagesLabel', { count: hiddenCount })}</VisuallyHidden>
        </Text>
      ) : null}
    </HStack>
  );
};

const PreviewStripTile = ({
  index,
  isSelected,
  name,
  onSelect,
}: {
  index: number;
  isSelected: boolean;
  name: string;
  onSelect: (index: number) => void;
}) => {
  const { t } = useTranslation();
  const handleClick = useCallback(() => onSelect(index), [index, onSelect]);

  return (
    <Box
      aria-label={t('widgets.gallery.export.previewImage', { index: index + 1 })}
      aria-pressed={isSelected}
      as="button"
      borderRadius="sm"
      boxSize="10"
      flexShrink={0}
      outline={isSelected ? '2px solid' : '1px solid'}
      outlineColor={isSelected ? 'accent.solid' : 'border.subtle'}
      outlineOffset={isSelected ? '1px' : '-1px'}
      overflow="hidden"
      _focusVisible={TILE_FOCUS_RING}
      onClick={handleClick}
    >
      <Image alt="" boxSize="full" draggable={false} objectFit="cover" src={galleryImageUrls.thumbnail(name)} />
    </Box>
  );
};

const ExportSummary = ({
  exportBytes,
  exportSize,
  isEncoding,
  isSelection,
  language,
  originalBytes,
}: {
  exportBytes: number | null;
  exportSize: ImageSize | null;
  isEncoding: boolean;
  isSelection: boolean;
  language: string;
  originalBytes: number | null;
}) => {
  const { t } = useTranslation();

  if (exportBytes === null) {
    return (
      <Text color="fg.muted" fontSize="sm">
        {isEncoding ? t('widgets.gallery.export.encoding') : ''}
      </Text>
    );
  }

  const size = formatBytes(exportBytes);
  const dimensions = exportSize
    ? { height: formatCount(exportSize.height), width: formatCount(exportSize.width) }
    : null;
  const change =
    originalBytes === null
      ? null
      : {
          change: new Intl.NumberFormat(language, {
            maximumFractionDigits: 0,
            signDisplay: 'exceptZero',
            style: 'percent',
          }).format(exportBytes / originalBytes - 1),
          original: formatBytes(originalBytes),
        };
  const summary =
    change && dimensions
      ? t('widgets.gallery.export.outputWithChangeAndDimensions', { ...change, ...dimensions, size })
      : change
        ? t('widgets.gallery.export.outputWithChange', { ...change, size })
        : dimensions
          ? t('widgets.gallery.export.outputWithDimensions', { ...dimensions, size })
          : size;

  return (
    <Text
      color="fg.muted"
      fontSize="sm"
      fontVariantNumeric="tabular-nums"
      opacity={isEncoding ? 0.6 : 1}
      overflow="hidden"
      textOverflow="ellipsis"
      whiteSpace="nowrap"
    >
      {isSelection ? t('widgets.gallery.export.previewedImageSummary', { summary }) : summary}
    </Text>
  );
};
