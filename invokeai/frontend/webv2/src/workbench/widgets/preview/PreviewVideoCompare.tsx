import type { GalleryVideoItem } from '@features/gallery/contracts';
import type { ComponentProps, MouseEvent } from 'react';
import type {
  VideoComparePlayer,
  VideoCompareSnapshot,
  VideoCompareStyles,
  VideoCompareTheme,
} from 'video-compare-player';

import { Box, Center, Flex, HStack, Spinner, Stack, Text } from '@chakra-ui/react';
import { useMountEffect } from '@platform/react/useMountEffect';
import { createExternalStoreCore, type ExternalStoreCore } from '@platform/state/externalStoreCore';
import { Button, IconButton } from '@platform/ui/Button';
import { Slider } from '@platform/ui/Slider';
import { Tooltip } from '@platform/ui/Tooltip';
import { formatMediaTime } from '@platform/ui/video-player/videoTimeline';
import { releaseFocusOnRemoval } from '@workbench/focusRegions';
import { MaximizeIcon, MinimizeIcon, PauseIcon, PlayIcon, Volume2Icon, VolumeXIcon, XIcon } from 'lucide-react';
import { useCallback, useMemo, useRef, useState, useSyncExternalStore } from 'react';
import { useTranslation } from 'react-i18next';
import { createVideoComparePlayer } from 'video-compare-player';
import 'video-compare-player/style.css';

import type { PreviewDensity } from './previewDensity';

import { previewStageStore } from './previewHeaderStore';
import { getPreviewStagePadding, PreviewStage } from './PreviewStage';

const attachStage = (node: HTMLDivElement | null) => {
  previewStageStore.setStageElement(node);

  if (!node) {
    return;
  }

  return () => {
    const focused = node.ownerDocument.activeElement;

    if (focused instanceof HTMLElement && node.contains(focused)) {
      releaseFocusOnRemoval(focused)();
    }

    previewStageStore.setStageElement(null);
  };
};

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
const CLIP_LABEL_STYLE = {
  background: 'var(--chakra-colors-bg-panel)',
  border: '1px solid var(--chakra-colors-border)',
  borderRadius: 'var(--chakra-radii-control)',
  fontSize: 'var(--chakra-font-sizes-xs)',
};
const COMPARE_STYLES = {
  divider: { transform: 'translateX(-50%)', width: '2px' },
  enhancedLabel: CLIP_LABEL_STYLE,
  handleVisual: {
    background: 'var(--chakra-colors-bg-panel)',
    border: '0',
    borderRadius: 'var(--chakra-radii-control)',
    boxShadow: 'none',
  },
  originalLabel: CLIP_LABEL_STYLE,
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
  '& [data-part="track"]': { bg: 'border.emphasized' },
} as const;
const COMPARE_CONTROLS_CSS = {
  '@container (max-width: 17rem)': { '& [data-compare-volume]': { display: 'none' } },
} as const;

interface CompareClips {
  comparison: GalleryVideoItem;
  primary: GalleryVideoItem;
}

type ControlsSnapshot = Pick<VideoCompareSnapshot, 'currentTime' | 'duration' | 'muted' | 'state' | 'volume'>;
type OnSliderValueChange = NonNullable<ComponentProps<typeof Slider>['onValueChange']>;
const INITIAL_CONTROLS: ControlsSnapshot = { currentTime: 0, duration: 0, muted: false, state: 'idle', volume: 1 };

const ComparisonControls = ({
  disabled,
  fullscreen,
  store,
  onSeek,
  onSetVolume,
  onToggleFullscreen,
  onToggleMute,
  onTogglePlayback,
}: {
  disabled: boolean;
  fullscreen: boolean;
  store: ExternalStoreCore<ControlsSnapshot>;
  onSeek: OnSliderValueChange;
  onSetVolume: OnSliderValueChange;
  onToggleFullscreen?: () => void;
  onToggleMute: () => void;
  onTogglePlayback: () => void;
}) => {
  const { t } = useTranslation();
  const snapshot = useSyncExternalStore(store.subscribe, store.getSnapshot, store.getSnapshot);

  const { currentTime, duration, muted, state, volume } = snapshot;
  const playing = state === 'playing' || state === 'buffering';
  const playLabel = t(playing ? 'common.videoPlayer.pause' : 'common.videoPlayer.play');
  const volumeLabel = t(muted ? 'common.videoPlayer.unmute' : 'common.videoPlayer.mute');
  const fullscreenLabel = t(fullscreen ? 'common.videoPlayer.exitFullscreen' : 'common.videoPlayer.fullscreen');
  const time = formatMediaTime(currentTime);
  const totalTime = formatMediaTime(duration);

  const seekLabel = useMemo(() => [t('common.videoPlayer.seek')], [t]);
  const seekValue = useMemo(() => [currentTime], [currentTime]);
  const volumeSliderLabel = useMemo(() => [t('common.videoPlayer.volume')], [t]);
  const volumeValue = useMemo(() => [muted ? 0 : volume], [muted, volume]);
  return (
    <HStack
      bg="bg.panel"
      borderColor="border"
      borderTopWidth="1px"
      css={COMPARE_CONTROLS_CSS}
      gap="1"
      minW="0"
      px="2"
      py="1"
    >
      <Tooltip content={playLabel} placement="top" portalled={!fullscreen}>
        <IconButton
          aria-label={playLabel}
          disabled={disabled}
          flexShrink={0}
          size="sm"
          variant="ghost"
          onClick={onTogglePlayback}
        >
          {playing ? <PauseIcon /> : <PlayIcon />}
        </IconButton>
      </Tooltip>
      <Slider
        aria-label={seekLabel}
        colorPalette="accent"
        css={COMPARE_SLIDER_CSS}
        disabled={disabled}
        flex="1"
        max={duration || 1}
        minW="0"
        size="md"
        step={0.1}
        value={seekValue}
        withThumbTooltip={false}
        onValueChange={onSeek}
      />
      <Text color="fg.muted" fontSize="xs" fontVariantNumeric="tabular-nums" whiteSpace="nowrap">
        {time} / {totalTime}
      </Text>
      <Tooltip content={volumeLabel} placement="top" portalled={!fullscreen}>
        <IconButton
          aria-label={volumeLabel}
          disabled={disabled}
          flexShrink={0}
          size="sm"
          variant="ghost"
          onClick={onToggleMute}
        >
          {muted ? <VolumeXIcon /> : <Volume2Icon />}
        </IconButton>
      </Tooltip>
      <Slider
        aria-label={volumeSliderLabel}
        css={COMPARE_SLIDER_CSS}
        data-compare-volume=""
        disabled={disabled}
        flexShrink={0}
        max={1}
        size="md"
        step={0.01}
        value={volumeValue}
        w="12"
        withThumbTooltip={false}
        onValueChange={onSetVolume}
      />
      {onToggleFullscreen && (
        <Tooltip content={fullscreenLabel} placement="top" portalled={!fullscreen}>
          <IconButton
            aria-label={fullscreenLabel}
            flexShrink={0}
            size="sm"
            variant="ghost"
            onClick={onToggleFullscreen}
          >
            {fullscreen ? <MinimizeIcon /> : <MaximizeIcon />}
          </IconButton>
        </Tooltip>
      )}
    </HStack>
  );
};

const ComparisonPlayback = ({
  comparison,
  primary,
  onRetry,
}: CompareClips & { onRetry: (event: MouseEvent<HTMLButtonElement>) => void }) => {
  const { t } = useTranslation();
  const frameRef = useRef<HTMLDivElement>(null);
  const hostRef = useRef<HTMLDivElement>(null);
  const playerRef = useRef<VideoComparePlayer | null>(null);
  const [controlsStore] = useState(() => createExternalStoreCore(INITIAL_CONTROLS));
  const [isReady, setIsReady] = useState(false);
  const [hasFailed, setHasFailed] = useState(false);
  const [fullscreen, setFullscreen] = useState(false);
  const label = t('widgets.preview.compareVideos');

  useMountEffect(() => {
    const frame = frameRef.current;
    const host = hostRef.current;

    if (!frame || !host) {
      return;
    }

    const owner = host.ownerDocument;
    let active = true;
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
        labels: true,
        modeSwitch: false,
      },
      enhancedPoster: primary.thumbnailUrl,
      enhancedSrc: primary.fullUrl,
      height: '100%',
      injectStyles: false,
      labels: {
        compareMode: label,
        dragHandle: t('widgets.preview.videoCompareDivider'),
        enhanced: `${primary.width} × ${primary.height} · ${primary.name}`,
        original: `${comparison.width} × ${comparison.height} · ${comparison.name}`,
      },
      loop: false,
      originalPoster: comparison.thumbnailUrl,
      originalSrc: comparison.fullUrl,
      preload: 'metadata',
      styles: COMPARE_STYLES,
      theme: COMPARE_THEME,
      onError: () => {
        if (!active) {
          return;
        }
        instance?.pause();
        if (frame.contains(owner.activeElement)) {
          frame.focus();
        }
        setHasFailed(true);
      },
      onReady: () => {
        if (active) {
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
      owner.removeEventListener('fullscreenchange', onFullscreenChange);
      unsubscribe();
      player.destroy();
    };
  });

  const togglePlayback = useCallback(() => void playerRef.current?.toggle().catch(() => undefined), []);
  const toggleMute = useCallback(() => {
    const player = playerRef.current;
    if (player) {
      player.setMuted(!player.getSnapshot().muted);
    }
  }, []);
  const seek = useCallback<OnSliderValueChange>(
    ({ value: [seconds] }) => {
      const player = playerRef.current;
      if (player && seconds !== undefined) {
        player.seek(seconds);
        controlsStore.patchSnapshot({ currentTime: player.getSnapshot().currentTime });
      }
    },
    [controlsStore]
  );
  const setVolume = useCallback<OnSliderValueChange>(({ value: [volume] }) => {
    if (volume !== undefined) {
      playerRef.current?.setVolume(volume);
    }
  }, []);

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

  return (
    <Flex
      ref={frameRef}
      aria-label={label}
      bg="bg"
      borderColor="border.emphasized"
      borderWidth="1px"
      containerType="inline-size"
      css={COMPARE_FRAME_CSS}
      direction="column"
      flex="1"
      minH="0"
      overflow="hidden"
      position="relative"
      role="group"
      rounded="md"
      tabIndex={-1}
    >
      <Box flex="1" minH="0" position="relative">
        <Box
          ref={hostRef}
          data-compare-loading={!isReady ? '' : undefined}
          inert={hasFailed}
          inset="0"
          position="absolute"
        />
        {hasFailed ? (
          <Center bg="bg.panel" inset="0" position="absolute" role="alert" zIndex="1">
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
          <Center inset="0" pointerEvents="none" position="absolute" role="status" zIndex="1">
            <HStack bg="bg.panel" borderColor="border" borderWidth="1px" gap="2" px="3" py="2" rounded="md">
              <Spinner aria-hidden="true" color="accent.solid" size="md" />
              <Text color="fg" fontSize="sm">
                {t('common.loading')}
              </Text>
            </HStack>
          </Center>
        ) : null}
      </Box>
      <ComparisonControls
        disabled={!isReady || hasFailed}
        fullscreen={fullscreen}
        store={controlsStore}
        onSeek={seek}
        onSetVolume={setVolume}
        onToggleFullscreen={document.fullscreenEnabled ? toggleFullscreen : undefined}
        onToggleMute={toggleMute}
        onTogglePlayback={togglePlayback}
      />
    </Flex>
  );
};

export const PreviewVideoCompare = ({
  comparison,
  density,
  primary,
  onExit,
}: CompareClips & { density: PreviewDensity; onExit: () => void }) => {
  const { i18n, t } = useTranslation();
  const [attempt, setAttempt] = useState(0);
  const exitRef = useRef<HTMLButtonElement>(null);
  const retry = useCallback((event: MouseEvent<HTMLButtonElement>) => {
    if (event.currentTarget === event.currentTarget.ownerDocument.activeElement) {
      exitRef.current?.focus({ preventScroll: true });
    }

    setAttempt((previous) => previous + 1);
  }, []);
  const exitLabel = t('widgets.preview.exitCompare');

  return (
    <PreviewStage ref={attachStage} fill="flex" padding={getPreviewStagePadding(density)}>
      <Flex direction="column" gap="2" h="full" minH="0" w="full">
        <HStack justify="space-between" minW="0">
          <Text color="fg" fontSize="md" fontWeight="semibold" truncate>
            {t('widgets.preview.compareVideos')}
          </Text>
          <IconButton
            ref={exitRef}
            aria-label={exitLabel}
            size="sm"
            tooltip={exitLabel}
            variant="ghost"
            onClick={onExit}
          >
            <XIcon aria-hidden="true" size={16} />
          </IconButton>
        </HStack>
        <ComparisonPlayback
          key={`${i18n.resolvedLanguage ?? i18n.language}:${attempt}`}
          comparison={comparison}
          primary={primary}
          onRetry={retry}
        />
      </Flex>
    </PreviewStage>
  );
};
