import type { GeneratedImageContract } from '@features/gallery';
import type { WidgetRuntimeApi } from '@workbench/widgetContracts';

import { Badge, Box, Flex, HStack, SegmentGroup, Stack, type SystemStyleObject } from '@chakra-ui/react';
import { Button } from '@platform/ui';
import { ArrowLeftRightIcon, XIcon } from 'lucide-react';
import {
  useCallback,
  useEffect,
  useEffectEvent,
  useMemo,
  useRef,
  useState,
  type FocusEvent,
  type PointerEvent,
} from 'react';
import { useTranslation } from 'react-i18next';

import type { PreviewComparisonMode } from './previewSettings';

import { useCompareLoupe, type CompareLoupePane } from './useCompareLoupe';

const COMPARISON_MODES: { labelKey: string; value: PreviewComparisonMode }[] = [
  { labelKey: 'widgets.preview.slider', value: 'slider' },
  { labelKey: 'widgets.preview.sideBySide', value: 'side-by-side' },
  { labelKey: 'widgets.preview.hover', value: 'hover' },
];

export const getNextPreviewComparisonMode = (mode: PreviewComparisonMode): PreviewComparisonMode => {
  const index = COMPARISON_MODES.findIndex((item) => item.value === mode);

  return COMPARISON_MODES[(index + 1) % COMPARISON_MODES.length]?.value ?? 'slider';
};

const getFittedFrameCss = (width: number, height: number): SystemStyleObject => ({
  aspectRatio: `${width} / ${height}`,
  height: 'auto',
  maxHeight: '100%',
  maxWidth: '100%',
  width: `min(100cqw, calc(100cqh * ${width / height}))`,
});

const BASE_IMAGE_STYLE = {
  display: 'block',
  height: 'auto',
  pointerEvents: 'none',
  userSelect: 'none',
  width: '100%',
} as const;

const OVERLAY_IMAGE_STYLE = {
  display: 'block',
  height: '100%',
  objectFit: 'contain',
  pointerEvents: 'none',
  userSelect: 'none',
  width: '100%',
} as const;

const previewGridCss = {
  backgroundImage: 'radial-gradient(circle, currentColor 1px, transparent 1.5px)',
  backgroundPosition: 'center',
  backgroundRepeat: 'repeat',
  backgroundSize: '24px 24px',
} as const;

const sliderTouchStyle = { touchAction: 'none' } as const;

/** Three-mode, project-persisted comparison surface. */
export const PreviewCompare = ({
  baseImage,
  compareImage,
  mode,
  onExit,
  onModeChange,
  onSwap,
  runtime,
}: {
  baseImage: GeneratedImageContract;
  compareImage: GeneratedImageContract;
  mode: PreviewComparisonMode;
  onExit: () => void;
  onModeChange: (mode: PreviewComparisonMode) => void;
  onSwap: () => void;
  runtime: WidgetRuntimeApi;
}) => {
  const { t } = useTranslation();
  const [dividerPercent, setDividerPercent] = useState(50);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const isDraggingRef = useRef(false);
  // Zoom-synced side-by-side: one shared transform drives both panes. Gated to
  // matching-dimension pairs, where fraction-space sync is exact.
  const dimensionsMatch = baseImage.width === compareImage.width && baseImage.height === compareImage.height;
  const compareLoupe = useCompareLoupe({
    enabled: mode === 'side-by-side' && dimensionsMatch,
    naturalWidth: baseImage.width,
  });
  const nextMode = useEffectEvent(() => onModeChange(getNextPreviewComparisonMode(mode)));
  const nextComparisonModeLabel = t('widgets.preview.commands.nextComparisonMode');

  useEffect(() => {
    const command = runtime.commands.register({
      handler: nextMode,
      id: 'viewer.nextComparisonMode',
      title: nextComparisonModeLabel,
    });
    const hotkey = runtime.hotkeys.register({
      commandId: 'viewer.nextComparisonMode',
      defaultKeys: ['m'],
      id: 'viewer.nextComparisonMode',
      title: nextComparisonModeLabel,
    });

    return () => {
      command();
      hotkey();
    };
  }, [nextComparisonModeLabel, runtime.commands, runtime.hotkeys]);

  const updateDivider = useCallback((clientX: number) => {
    const rect = containerRef.current?.getBoundingClientRect();

    if (!rect || rect.width === 0) {
      return;
    }

    setDividerPercent(Math.min(100, Math.max(0, ((clientX - rect.left) / rect.width) * 100)));
  }, []);
  const handlePointerDown = useCallback(
    (event: PointerEvent<HTMLDivElement>) => {
      if (event.pointerType === 'mouse' && event.button !== 0) {
        return;
      }

      event.preventDefault();
      isDraggingRef.current = true;
      event.currentTarget.setPointerCapture(event.pointerId);
      updateDivider(event.clientX);
    },
    [updateDivider]
  );
  const handlePointerMove = useCallback(
    (event: PointerEvent<HTMLDivElement>) => {
      if (isDraggingRef.current) {
        event.preventDefault();
        updateDivider(event.clientX);
      }
    },
    [updateDivider]
  );
  const endDrag = useCallback((event?: PointerEvent<HTMLDivElement>) => {
    if (event?.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }

    isDraggingRef.current = false;
  }, []);
  const handleModeChange = useCallback(
    (details: { value: string | null }) => {
      if (details.value === 'slider' || details.value === 'side-by-side' || details.value === 'hover') {
        onModeChange(details.value);
      }
    },
    [onModeChange]
  );
  const clipStyle = useMemo(() => ({ clipPath: `inset(0 ${100 - dividerPercent}% 0 0)` }), [dividerPercent]);

  return (
    <Stack gap="3" h="full" minH="0" w="full">
      <Flex
        align="center"
        backgroundColor="bg.inset"
        borderWidth="1px"
        borderColor="border.subtle"
        color="fg.grid"
        containerType="size"
        css={previewGridCss}
        flex="1"
        justify="center"
        minH="0"
        overflow="hidden"
        p="6"
        rounded="lg"
        w="full"
      >
        {mode === 'slider' ? (
          <Box
            ref={containerRef}
            borderColor="border.emphasized"
            borderWidth="1px"
            cursor="ew-resize"
            css={getFittedFrameCss(baseImage.width, baseImage.height)}
            overflow="hidden"
            position="relative"
            rounded="lg"
            style={sliderTouchStyle}
            onLostPointerCapture={endDrag}
            onPointerCancel={endDrag}
            onPointerDown={handlePointerDown}
            onPointerMove={handlePointerMove}
            onPointerUp={endDrag}
          >
            <img
              alt={baseImage.imageName}
              draggable={false}
              height={baseImage.height}
              src={baseImage.imageUrl}
              style={BASE_IMAGE_STYLE}
              width={baseImage.width}
            />
            <Box bg="transparent" inset="0" pointerEvents="none" position="absolute" style={clipStyle}>
              <img
                alt={compareImage.imageName}
                draggable={false}
                height={compareImage.height}
                src={compareImage.imageUrl}
                style={OVERLAY_IMAGE_STYLE}
                width={compareImage.width}
              />
            </Box>
            <Box
              bg="accent.solid"
              bottom="0"
              left={`${dividerPercent}%`}
              pointerEvents="none"
              position="absolute"
              top="0"
              transform="translateX(-50%)"
              w="2px"
            />
            <ComparisonBadges compareLabel={t('widgets.preview.compare')} viewingLabel={t('widgets.preview.viewing')} />
          </Box>
        ) : mode === 'hover' ? (
          <HoverCompareFrame baseImage={baseImage} compareImage={compareImage} />
        ) : (
          <HStack gap="2" h="full" minH="0" w="full">
            <CompareSidePane
              image={compareImage}
              isZoomed={compareLoupe.isZoomed}
              label={t('widgets.preview.compare')}
              pane={compareLoupe.getPane(0)}
            />
            <CompareSidePane
              image={baseImage}
              isZoomed={compareLoupe.isZoomed}
              label={t('widgets.preview.viewing')}
              pane={compareLoupe.getPane(1)}
            />
          </HStack>
        )}
      </Flex>
      <HStack flexShrink={0} gap="1" justify="center">
        <SegmentGroup.Root size="xs" value={mode} onValueChange={handleModeChange}>
          <SegmentGroup.Indicator />
          {COMPARISON_MODES.map((item) => (
            <SegmentGroup.Item key={item.value} value={item.value}>
              <SegmentGroup.ItemHiddenInput />
              <SegmentGroup.ItemText>{t(item.labelKey)}</SegmentGroup.ItemText>
            </SegmentGroup.Item>
          ))}
        </SegmentGroup.Root>
        <Button size="2xs" variant="outline" onClick={onSwap}>
          <ArrowLeftRightIcon />
          {t('common.swap')}
        </Button>
        <Button size="2xs" variant="outline" onClick={onExit}>
          <XIcon />
          {t('widgets.preview.exitCompare')}
        </Button>
      </HStack>
    </Stack>
  );
};

const HoverCompareFrame = ({
  baseImage,
  compareImage,
}: {
  baseImage: GeneratedImageContract;
  compareImage: GeneratedImageContract;
}) => {
  const { t } = useTranslation();
  const [isRevealed, setIsRevealed] = useState(false);
  const interactionRef = useRef({ focused: false, hovered: false, pressed: false });
  const syncRevealed = useCallback(() => {
    const interaction = interactionRef.current;

    setIsRevealed(interaction.focused || interaction.hovered || interaction.pressed);
  }, []);
  const handleFocus = useCallback(
    (_event: FocusEvent<HTMLDivElement>) => {
      interactionRef.current.focused = true;
      syncRevealed();
    },
    [syncRevealed]
  );
  const handleBlur = useCallback(
    (_event: FocusEvent<HTMLDivElement>) => {
      interactionRef.current.focused = false;
      syncRevealed();
    },
    [syncRevealed]
  );
  const handlePointerEnter = useCallback(
    (event: PointerEvent<HTMLDivElement>) => {
      if (event.pointerType === 'mouse') {
        interactionRef.current.hovered = true;
        syncRevealed();
      }
    },
    [syncRevealed]
  );
  const handlePointerLeave = useCallback(
    (event: PointerEvent<HTMLDivElement>) => {
      if (event.pointerType === 'mouse') {
        interactionRef.current.hovered = false;
        syncRevealed();
      }
    },
    [syncRevealed]
  );
  const handlePointerDown = useCallback(
    (event: PointerEvent<HTMLDivElement>) => {
      if (event.pointerType !== 'mouse') {
        event.preventDefault();
        interactionRef.current.pressed = true;
        event.currentTarget.setPointerCapture(event.pointerId);
        syncRevealed();
      }
    },
    [syncRevealed]
  );
  const handlePointerEnd = useCallback(
    (event: PointerEvent<HTMLDivElement>) => {
      if (event.pointerType !== 'mouse') {
        if (event.currentTarget.hasPointerCapture(event.pointerId)) {
          event.currentTarget.releasePointerCapture(event.pointerId);
        }
        interactionRef.current.pressed = false;
        syncRevealed();
      }
    },
    [syncRevealed]
  );

  return (
    <Box
      aria-label={t('widgets.preview.hoverComparisonAriaLabel')}
      borderColor="border.emphasized"
      borderWidth="1px"
      css={getFittedFrameCss(baseImage.width, baseImage.height)}
      cursor="pointer"
      overflow="hidden"
      position="relative"
      rounded="lg"
      tabIndex={0}
      touchAction="none"
      onBlur={handleBlur}
      onFocus={handleFocus}
      onLostPointerCapture={handlePointerEnd}
      onPointerCancel={handlePointerEnd}
      onPointerDown={handlePointerDown}
      onPointerEnter={handlePointerEnter}
      onPointerLeave={handlePointerLeave}
      onPointerUp={handlePointerEnd}
    >
      <img
        alt={baseImage.imageName}
        draggable={false}
        height={baseImage.height}
        src={baseImage.imageUrl}
        style={BASE_IMAGE_STYLE}
        width={baseImage.width}
      />
      <Box
        inset="0"
        opacity={isRevealed ? 1 : 0}
        pointerEvents="none"
        position="absolute"
        transitionDuration="var(--wb-motion-duration-fast)"
        transitionProperty="opacity"
        transitionTimingFunction="ease"
      >
        <img
          alt={compareImage.imageName}
          draggable={false}
          height={compareImage.height}
          src={compareImage.imageUrl}
          style={OVERLAY_IMAGE_STYLE}
          width={compareImage.width}
        />
      </Box>
      <ComparisonBadges compareLabel={t('widgets.preview.compare')} viewingLabel={t('widgets.preview.viewing')} />
    </Box>
  );
};

const ComparisonBadges = ({ compareLabel, viewingLabel }: { compareLabel: string; viewingLabel: string }) => (
  <>
    <Badge left="2" pointerEvents="none" position="absolute" size="xs" top="2" variant="solid">
      {compareLabel}
    </Badge>
    <Badge pointerEvents="none" position="absolute" right="2" size="xs" top="2" variant="solid">
      {viewingLabel}
    </Badge>
  </>
);

const CompareSidePane = ({
  image,
  isZoomed = false,
  label,
  pane = null,
}: {
  image: GeneratedImageContract;
  isZoomed?: boolean;
  label: string;
  /** Shared-loupe pane handle when zoom-synced comparison is active. */
  pane?: CompareLoupePane | null;
}) => (
  <Flex align="center" containerType="size" flex="1" h="full" justify="center" minW="0" overflow="hidden">
    <Box
      ref={pane?.frameRefCallback}
      borderColor="border.emphasized"
      borderWidth="1px"
      css={getFittedFrameCss(image.width, image.height)}
      cursor={isZoomed ? 'grab' : undefined}
      overflow="hidden"
      position="relative"
      rounded="lg"
      {...pane?.frameProps}
    >
      <img
        ref={pane?.imageRefCallback}
        alt={image.imageName}
        draggable={false}
        height={image.height}
        src={image.imageUrl}
        style={BASE_IMAGE_STYLE}
        width={image.width}
      />
      <Badge left="2" pointerEvents="none" position="absolute" size="xs" top="2" variant="solid">
        {label}
      </Badge>
    </Box>
  </Flex>
);
