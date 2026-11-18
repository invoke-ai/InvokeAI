import type { GalleryVideoItem } from '@features/gallery/contracts';
import type { LucideIcon } from 'lucide-react';
import type { ComponentProps, KeyboardEvent, MouseEvent } from 'react';
import type {
  VideoComparePlayer,
  VideoCompareSnapshot,
  VideoCompareStyles,
  VideoCompareTheme,
} from 'video-compare-player';

import { Box, Center, HStack, Icon, Kbd, Spinner, Stack, Text, type SystemStyleObject } from '@chakra-ui/react';
import { isImeComposing } from '@platform/browser/imeComposition';
import { useMountEffect } from '@platform/react/useMountEffect';
import { createExternalStoreCore, type ExternalStoreCore } from '@platform/state/externalStoreCore';
import { Button, IconButton } from '@platform/ui/Button';
import { Slider } from '@platform/ui/Slider';
import { Tooltip } from '@platform/ui/Tooltip';
import { formatMediaTime, getFrameAtTime, getFrameCount, getFrameTime } from '@platform/ui/video-player/videoTimeline';
import { releaseFocusOnRemoval } from '@workbench/focusRegions';
import {
  MaximizeIcon,
  MinimizeIcon,
  PauseIcon,
  PlayIcon,
  StepBackIcon,
  StepForwardIcon,
  Volume2Icon,
  VolumeXIcon,
} from 'lucide-react';
import { useCallback, useMemo, useRef, useState, useSyncExternalStore } from 'react';
import { useTranslation } from 'react-i18next';
import { createVideoComparePlayer } from 'video-compare-player';
import 'video-compare-player/style.css';

import type { PreviewDensity } from './previewDensity';

import { PreviewComparisonBadges, PreviewComparisonDock } from './PreviewCompare';
import { previewStageStore } from './previewHeaderStore';
import { getFittedFrameCss, getPreviewStagePadding, PreviewStage } from './PreviewStage';

const { setStageElement } = previewStageStore;

const attachComparison = (node: HTMLDivElement | null) =>
  node
    ? () => {
        const focused = node.ownerDocument.activeElement;
        if (focused instanceof HTMLElement && node.contains(focused)) {
          releaseFocusOnRemoval(focused)();
        }
      }
    : undefined;

const COMPARISON_FULLSCREEN_CSS = {
  '&:fullscreen': {
    bg: 'black',
    borderWidth: '0',
    h: 'full',
    w: 'full',
  },
} satisfies SystemStyleObject;

// CSS variables keep the mounted player in sync with theme and contrast changes.
const COMPARE_THEME = {
  accentColor: 'var(--chakra-colors-accent-solid)',
  backgroundColor: 'var(--chakra-colors-bg)',
  controlRadius: 'var(--chakra-radii-control)',
  dividerColor: 'var(--chakra-colors-accent-solid)',
  dividerShadow: 'none',
  fontFamily: 'var(--chakra-fonts-body)',
  textColor: 'var(--chakra-colors-fg)',
} satisfies Partial<VideoCompareTheme>;
const COMPARE_STYLES = {
  divider: { transform: 'translateX(-50%)', width: '2px' },
  handleVisual: {
    background: 'var(--chakra-colors-bg-panel)',
    border: '0',
    borderRadius: 'var(--chakra-radii-control)',
    boxShadow: 'none',
  },
  originalLayer: { background: 'var(--chakra-colors-bg)' },
  root: { minHeight: '0' },
  spinner: {
    borderColor: 'var(--chakra-colors-border)',
    borderTopColor: 'var(--chakra-colors-accent-solid)',
  },
} satisfies VideoCompareStyles;
const COMPARE_FRAME_CSS = {
  '& [part="handle-visual"]': {
    outline: '1px solid {colors.accent.solid}',
    outlineOffset: '-1px',
  },
  '& button[part="handle"]:focus-visible': {
    outline: 'none',
    '& [part="handle-visual"]': {
      outlineWidth: '2px',
      outlineOffset: '-2px',
    },
  },
  '& [part="spinner"]': { _motionReduce: { animation: 'none' } },
  '& [data-compare-loading] [part="loading"]': { visibility: 'hidden' },
};
const COMPARE_SLIDER_CSS = {
  '--slider-track-size': 'sizes.1',
  '& [data-part="track"]': { bg: 'whiteAlpha.300' },
} as const;
const PLAYER_CONTAINER = 'comparison-player';
const FRAME_READOUT_CSS = { [`@container ${PLAYER_CONTAINER} (width < 370px)`]: { display: 'none' } };
const PLAYBACK_BUTTON_CSS = {
  [`@container ${PLAYER_CONTAINER} (width < 250px)`]: { h: '6', minW: '6', w: '6' },
};
const COMPARE_VOLUME_CSS = {
  ...COMPARE_SLIDER_CSS,
  [`@container ${PLAYER_CONTAINER} (width < 17rem)`]: { display: 'none' },
};
const SCRIM_BUTTON_HOVER = { bg: 'whiteAlpha.200' } as const;

type PlaybackBookmark = Pick<VideoCompareSnapshot, 'currentTime' | 'muted' | 'position' | 'volume'>;

interface CompareClips {
  comparison: GalleryVideoItem;
  primary: GalleryVideoItem;
}

type ControlsSnapshot = Pick<VideoCompareSnapshot, 'currentTime' | 'duration' | 'muted' | 'state' | 'volume'>;
type OnSliderValueChange = NonNullable<ComponentProps<typeof Slider>['onValueChange']>;
const INITIAL_CONTROLS: ControlsSnapshot = { currentTime: 0, duration: 0, muted: false, state: 'idle', volume: 1 };

const PlaybackButton = ({
  disabled,
  fullscreen,
  icon,
  label,
  shortcut,
  onClick,
}: {
  disabled?: boolean;
  fullscreen: boolean;
  icon: LucideIcon;
  label: string;
  shortcut: string;
  onClick: () => void;
}) => {
  const tooltip = useMemo(
    () => (
      <HStack gap="1.5">
        <Text>{label}</Text>
        <Kbd variant="subtle">{shortcut}</Kbd>
      </HStack>
    ),
    [label, shortcut]
  );
  return (
    <Tooltip content={tooltip} placement="top" portalled={!fullscreen}>
      <IconButton
        _hover={SCRIM_BUTTON_HOVER}
        aria-label={label}
        aria-keyshortcuts={shortcut}
        color="white"
        css={PLAYBACK_BUTTON_CSS}
        disabled={disabled}
        flexShrink={0}
        pointerEvents="auto"
        size="sm"
        variant="ghost"
        onClick={onClick}
      >
        <Icon as={icon} boxSize="3.5" />
      </IconButton>
    </Tooltip>
  );
};

const ComparisonControls = ({
  disabled,
  fps,
  fullscreen,
  store,
  onSeek,
  onSetVolume,
  onStepFrame,
  onToggleFullscreen,
  onToggleMute,
  onTogglePlayback,
}: {
  disabled: boolean;
  fps: number | null;
  fullscreen: boolean;
  store: ExternalStoreCore<ControlsSnapshot>;
  onSeek: OnSliderValueChange;
  onSetVolume: OnSliderValueChange;
  onStepFrame: (direction: -1 | 1) => void;
  onToggleFullscreen?: () => void;
  onToggleMute: () => void;
  onTogglePlayback: () => void;
}) => {
  const { t } = useTranslation();
  const { currentTime, duration, muted, state, volume } = useSyncExternalStore(
    store.subscribe,
    store.getSnapshot,
    store.getSnapshot
  );
  const playing = state === 'playing' || state === 'buffering';
  const playLabel = t(playing ? 'common.videoPlayer.pause' : 'common.videoPlayer.play');
  const volumeLabel = t(muted ? 'common.videoPlayer.unmute' : 'common.videoPlayer.mute');
  const fullscreenLabel = t(fullscreen ? 'common.videoPlayer.exitFullscreen' : 'common.videoPlayer.fullscreen');
  const frameCount = fps === null ? 1 : getFrameCount(duration, fps);
  const frame = fps === null ? null : getFrameAtTime(currentTime, fps, frameCount);

  const seekLabel = useMemo(() => [t('common.videoPlayer.seek')], [t]);
  const seekValue = useMemo(() => [currentTime], [currentTime]);
  const volumeSliderLabel = useMemo(() => [t('common.videoPlayer.volume')], [t]);
  const volumeValue = useMemo(() => [muted ? 0 : volume], [muted, volume]);
  const previousFrame = useCallback(() => onStepFrame(-1), [onStepFrame]);
  const nextFrame = useCallback(() => onStepFrame(1), [onStepFrame]);
  return (
    <Stack
      bgGradient="to-t"
      bottom="0"
      color="white"
      gap="2"
      gradientFrom="blackAlpha.800"
      gradientTo="transparent"
      insetX="0"
      pb="1.5"
      pointerEvents="none"
      position="absolute"
      pt="8"
      px="2"
      zIndex="2"
    >
      <HStack px="1.5">
        <Slider
          aria-label={seekLabel}
          colorPalette="accent"
          css={COMPARE_SLIDER_CSS}
          disabled={disabled}
          flex="1"
          max={duration || 1}
          minW="0"
          pointerEvents="auto"
          size="md"
          step={0.1}
          thumbAlignment="center"
          value={seekValue}
          withThumbTooltip={false}
          onValueChange={onSeek}
        />
      </HStack>
      <HStack gap="1" minW="0">
        <PlaybackButton
          disabled={disabled}
          fullscreen={fullscreen}
          icon={playing ? PauseIcon : PlayIcon}
          label={playLabel}
          shortcut="K"
          onClick={onTogglePlayback}
        />
        {fps === null ? null : (
          <HStack gap="1">
            <PlaybackButton
              disabled={disabled}
              fullscreen={fullscreen}
              icon={StepBackIcon}
              label={t('common.videoPlayer.previousFrame')}
              shortcut=","
              onClick={previousFrame}
            />
            <PlaybackButton
              disabled={disabled}
              fullscreen={fullscreen}
              icon={StepForwardIcon}
              label={t('common.videoPlayer.nextFrame')}
              shortcut="."
              onClick={nextFrame}
            />
          </HStack>
        )}
        <HStack
          color="whiteAlpha.900"
          fontSize="sm"
          fontVariantNumeric="tabular-nums"
          gap="2"
          minW="0"
          whiteSpace="nowrap"
        >
          <Text truncate>
            {formatMediaTime(currentTime)} / {formatMediaTime(duration)}
          </Text>
          {frame === null ? null : (
            <Text color="whiteAlpha.800" css={FRAME_READOUT_CSS} flexShrink={0}>
              {t('common.videoPlayer.frame', { frame })} / {frameCount - 1}
            </Text>
          )}
        </HStack>
        <Box flex="1" />
        <PlaybackButton
          disabled={disabled}
          fullscreen={fullscreen}
          icon={muted ? VolumeXIcon : Volume2Icon}
          label={volumeLabel}
          shortcut="M"
          onClick={onToggleMute}
        />
        <Slider
          aria-label={volumeSliderLabel}
          css={COMPARE_VOLUME_CSS}
          disabled={disabled}
          flexShrink={0}
          max={1}
          pointerEvents="auto"
          size="md"
          step={0.01}
          value={volumeValue}
          w="12"
          withThumbTooltip={false}
          onValueChange={onSetVolume}
        />
        {onToggleFullscreen ? (
          <PlaybackButton
            fullscreen={fullscreen}
            icon={fullscreen ? MinimizeIcon : MaximizeIcon}
            label={fullscreenLabel}
            shortcut="F"
            onClick={onToggleFullscreen}
          />
        ) : null}
      </HStack>
    </Stack>
  );
};

const ComparisonPlayback = ({
  comparison,
  primary,
  saveBookmark,
  getBookmark,
  onRetry,
}: CompareClips & {
  saveBookmark: (snapshot: PlaybackBookmark) => void;
  getBookmark: () => PlaybackBookmark | null;
  onRetry: (event: MouseEvent<HTMLButtonElement>) => void;
}) => {
  const { t } = useTranslation();
  const frameRef = useRef<HTMLDivElement>(null);
  const hostRef = useRef<HTMLDivElement>(null);
  const playerRef = useRef<VideoComparePlayer | null>(null);
  const [controlsStore] = useState(() => createExternalStoreCore(INITIAL_CONTROLS));
  const [isReady, setIsReady] = useState(false);
  const [hasFailed, setHasFailed] = useState(false);
  const [fullscreen, setFullscreen] = useState(false);
  const label = t('widgets.preview.compareVideos');
  const fps = primary.fps !== undefined && Number.isFinite(primary.fps) && primary.fps > 0 ? primary.fps : null;
  const frameCss = useMemo(
    () => ({
      ...getFittedFrameCss(primary.width, primary.height),
      ...COMPARE_FRAME_CSS,
      ...COMPARISON_FULLSCREEN_CSS,
    }),
    [primary.width, primary.height]
  );

  useMountEffect(() => {
    const frame = frameRef.current;
    const host = hostRef.current;

    if (!frame || !host) {
      return;
    }

    const owner = host.ownerDocument;
    const bookmark = getBookmark();
    let active = true;
    let ready = false;
    let failed = false;
    setHasFailed(false);
    setIsReady(false);
    setFullscreen(owner.fullscreenElement === frame);
    // The first selected asset remains the playback clock and the only source of audio.
    let instance: VideoComparePlayer | undefined;
    instance = createVideoComparePlayer(host, {
      aspectRatio: 'auto',
      attributes: {
        enhancedVideo: { 'aria-label': t('widgets.preview.videoLabel', { name: primary.name }) },
        originalVideo: { 'aria-label': t('widgets.preview.videoLabel', { name: comparison.name }) },
        root: { tabIndex: -1 },
      },
      controls: {
        centerPlayButton: false,
        clickToToggle: false,
        enabled: false,
        fullscreen: false,
        keyboard: false,
        labels: false,
        modeSwitch: false,
      },
      enhancedPoster: primary.thumbnailUrl,
      enhancedSrc: primary.fullUrl,
      height: '100%',
      ...(bookmark ? { initialPosition: bookmark.position, muted: bookmark.muted, volume: bookmark.volume } : {}),
      injectStyles: false,
      labels: {
        compareMode: label,
        dragHandle: t('widgets.preview.videoCompareDivider'),
      },
      loop: false,
      originalPoster: comparison.thumbnailUrl,
      originalSrc: comparison.fullUrl,
      preload: 'metadata',
      styles: COMPARE_STYLES,
      theme: COMPARE_THEME,
      onError: () => {
        if (!active || failed) {
          return;
        }
        failed = true;
        instance?.pause();
        if (frame.contains(owner.activeElement)) {
          frame.focus();
        }
        setHasFailed(true);
      },
      onReady: ({ player }) => {
        if (active && !failed) {
          ready = true;
          if (bookmark && bookmark.currentTime > 0) {
            player.seek(Math.min(bookmark.currentTime, player.getSnapshot().duration));
          }
          setIsReady(true);
        }
      },
    });
    const player = instance;
    playerRef.current = player;
    // React Activity disconnects both effects and keeps their last render. The controls subscribe
    // to this stable store and cannot reconnect to the old player while this effect recreates it.
    const unsubscribe = player.subscribe(({ currentTime, duration, muted, state, volume }) => {
      if (active) {
        controlsStore.setSnapshot({ currentTime, duration, muted, state, volume });
      }
    });
    const onFullscreenChange = () => setFullscreen(owner.fullscreenElement === frame);
    owner.addEventListener('fullscreenchange', onFullscreenChange);

    return () => {
      active = false;
      playerRef.current = null;
      // Loading or failed players cannot replace a ready bookmark, and Retry clears it before remount.
      if (ready && !failed) {
        saveBookmark(player.getSnapshot());
      }
      owner.removeEventListener('fullscreenchange', onFullscreenChange);
      unsubscribe();
      player.destroy();
    };
  });

  const togglePlayback = useCallback(() => void playerRef.current?.toggle().catch(() => undefined), [playerRef]);
  const toggleMute = useCallback(() => {
    const player = playerRef.current;
    if (player) {
      player.setMuted(!player.getSnapshot().muted);
    }
  }, [playerRef]);
  const seekTo = useCallback(
    (seconds: number) => {
      const player = playerRef.current;
      if (player) {
        player.seek(seconds);
        controlsStore.patchSnapshot({ currentTime: player.getSnapshot().currentTime });
      }
    },
    [controlsStore, playerRef]
  );
  const seek = useCallback<OnSliderValueChange>(
    ({ value: [seconds] }) => {
      if (seconds !== undefined) {
        seekTo(seconds);
      }
    },
    [seekTo]
  );
  const stepFrame = useCallback(
    (direction: -1 | 1) => {
      const player = playerRef.current;
      if (!player || fps === null) {
        return;
      }

      const { currentTime, duration } = player.getSnapshot();
      const count = getFrameCount(duration, fps);
      const frame = Math.min(count - 1, Math.max(0, getFrameAtTime(currentTime, fps, count) + direction));
      player.pause();
      seekTo(getFrameTime(frame, fps));
    },
    [fps, playerRef, seekTo]
  );
  const setVolume = useCallback<OnSliderValueChange>(
    ({ value: [volume] }) => {
      if (volume !== undefined) {
        playerRef.current?.setVolume(volume);
      }
    },
    [playerRef]
  );

  const toggleFullscreen = useCallback(() => {
    const frame = frameRef.current;
    if (!frame) {
      return;
    }
    if (frame.ownerDocument.fullscreenElement === frame) {
      void frame.ownerDocument.exitFullscreen().catch(() => undefined);
    } else {
      void frame.requestFullscreen().catch(() => undefined);
    }
  }, []);

  const controlsDisabled = !isReady || hasFailed;
  const handleKeyDown = useCallback(
    (event: KeyboardEvent<HTMLDivElement>) => {
      const target = event.target;
      if (
        event.defaultPrevented ||
        event.altKey ||
        event.ctrlKey ||
        event.metaKey ||
        event.shiftKey ||
        isImeComposing(event.nativeEvent) ||
        !(target instanceof Element) ||
        !event.currentTarget.contains(target) ||
        target.closest('input, textarea, select, [contenteditable]:not([contenteditable="false"])')
      ) {
        return;
      }

      const key = event.key.toLowerCase();
      if (key === ' ' && target.closest('button, a[href], [role="button"]')) {
        event.stopPropagation();
        return;
      }

      switch (key) {
        case ',':
        case '.':
          if (!controlsDisabled) {
            stepFrame(key === ',' ? -1 : 1);
          }
          break;
        case ' ':
        case 'k':
          if (!controlsDisabled) {
            togglePlayback();
          }
          break;
        case 'm':
          if (!controlsDisabled) {
            toggleMute();
          }
          break;
        case 'f':
          if (!hasFailed && document.fullscreenEnabled) {
            toggleFullscreen();
          }
          break;
        default:
          return;
      }
      event.preventDefault();
      event.stopPropagation();
    },
    [controlsDisabled, hasFailed, stepFrame, toggleFullscreen, toggleMute, togglePlayback]
  );

  return (
    <Box
      ref={frameRef}
      aria-label={label}
      bg="bg"
      borderColor="border.emphasized"
      borderWidth="1px"
      containerName={PLAYER_CONTAINER}
      containerType="inline-size"
      css={frameCss}
      minH="min(100cqh, 110px)"
      minW="min(100cqw, 176px)"
      overflow="hidden"
      position="relative"
      role="group"
      tabIndex={0}
      onKeyDown={handleKeyDown}
    >
      <Box
        ref={hostRef}
        data-compare-loading={!isReady ? '' : undefined}
        inert={hasFailed}
        inset="0"
        position="absolute"
      />
      {isReady && !hasFailed ? (
        // The package reveals the follower at the left of its divider.
        <PreviewComparisonBadges
          leftLabel={`${t('widgets.preview.viewing')} · ${comparison.width} × ${comparison.height}`}
          rightLabel={`${t('widgets.preview.compare')} · ${primary.width} × ${primary.height}`}
        />
      ) : null}
      {hasFailed ? (
        <Center bg="bg.panel" inset="0" position="absolute" role="alert" zIndex="3">
          <Stack align="center" gap="2">
            <Text color="fg" fontSize="lg">
              {t('widgets.preview.videoFailed')}
            </Text>
            <Button size="sm" onClick={onRetry}>
              {t('widgets.preview.videoRetry')}
            </Button>
          </Stack>
        </Center>
      ) : !isReady ? (
        <Center inset="0" pointerEvents="none" position="absolute" role="status" zIndex="3">
          <HStack bg="bg.panel" borderColor="border" borderWidth="1px" gap="2" px="3" py="2" rounded="md">
            <Spinner aria-hidden="true" color="accent.solid" size="md" />
            <Text color="fg" fontSize="sm">
              {t('common.loading')}
            </Text>
          </HStack>
        </Center>
      ) : null}
      {hasFailed ? null : (
        <ComparisonControls
          disabled={controlsDisabled}
          fps={fps}
          fullscreen={fullscreen}
          store={controlsStore}
          onSeek={seek}
          onSetVolume={setVolume}
          onStepFrame={stepFrame}
          onToggleFullscreen={document.fullscreenEnabled ? toggleFullscreen : undefined}
          onToggleMute={toggleMute}
          onTogglePlayback={togglePlayback}
        />
      )}
    </Box>
  );
};

export const PreviewVideoCompare = ({
  comparison,
  density,
  primary,
  onExit,
  onSwap,
}: CompareClips & { density: PreviewDensity; onExit: () => void; onSwap: () => void }) => {
  const { i18n } = useTranslation();
  const [attempt, setAttempt] = useState(0);
  const exitRef = useRef<HTMLButtonElement>(null);
  const playbackBookmark = useRef<PlaybackBookmark | null>(null);
  const saveBookmark = useCallback(({ currentTime, muted, position, volume }: PlaybackBookmark) => {
    playbackBookmark.current = { currentTime, muted, position, volume };
  }, []);
  const getBookmark = useCallback(() => playbackBookmark.current, []);
  const retry = useCallback((event: MouseEvent<HTMLButtonElement>) => {
    if (event.currentTarget === event.currentTarget.ownerDocument.activeElement) {
      exitRef.current?.focus({ preventScroll: true });
    }

    playbackBookmark.current = null;
    setAttempt((previous) => previous + 1);
  }, []);
  return (
    <Stack ref={attachComparison} gap="0" h="full" minH="0" w="full">
      <PreviewStage ref={setStageElement} fill="flex" padding={getPreviewStagePadding(density)}>
        <ComparisonPlayback
          key={JSON.stringify([
            i18n.resolvedLanguage ?? i18n.language,
            attempt,
            primary.name,
            primary.fullUrl,
            comparison.fullUrl,
          ])}
          comparison={comparison}
          primary={primary}
          saveBookmark={saveBookmark}
          getBookmark={getBookmark}
          onRetry={retry}
        />
      </PreviewStage>
      <PreviewComparisonDock density={density} exitRef={exitRef} onExit={onExit} onSwap={onSwap} />
    </Stack>
  );
};
