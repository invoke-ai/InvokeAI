import type { LucideIcon } from 'lucide-react';

import {
  Box,
  Center,
  chakra,
  HStack,
  Icon,
  Kbd,
  Menu,
  mergeRefs,
  Portal,
  Spinner,
  Stack,
  Text,
} from '@chakra-ui/react';
import { isImeComposing } from '@platform/browser/imeComposition';
import { useMountEffect } from '@platform/react/useMountEffect';
import { useExternalStoreSelector } from '@platform/state/selectors';
import { Button, IconButton } from '@platform/ui/Button';
import { MenuContent } from '@platform/ui/Menu';
import { Slider } from '@platform/ui/Slider';
import { Tooltip, useTooltipTriggerIds } from '@platform/ui/Tooltip';
import {
  MaximizeIcon,
  MinimizeIcon,
  PauseIcon,
  PlayIcon,
  RepeatIcon,
  StepBackIcon,
  StepForwardIcon,
  Volume1Icon,
  Volume2Icon,
  VolumeXIcon,
} from 'lucide-react';
import {
  useCallback,
  useId,
  useMemo,
  useRef,
  useState,
  useSyncExternalStore,
  type CSSProperties,
  type KeyboardEvent,
  type MouseEvent,
  type PointerEvent,
  type ReactNode,
  type Ref,
  type SyntheticEvent,
  type VideoHTMLAttributes,
} from 'react';
import { useTranslation } from 'react-i18next';

import { createMediaElementStore, type MediaElementSnapshot, type MediaElementStore } from './mediaElementStore';
import {
  clampLoopRange,
  formatMediaTime,
  getFrameAtTime,
  getFrameCount,
  getFrameTime,
  getScrubSteps,
  type LoopRange,
} from './videoTimeline';

export interface VideoPlayerProps extends Omit<
  VideoHTMLAttributes<HTMLVideoElement>,
  'aria-label' | 'children' | 'controls' | 'loop' | 'src'
> {
  /** Names the player. */
  'aria-label': string;
  src: string;
  /** Enables frame stepping and the frame readout. */
  fps?: number;
  /**
   * The clip's frames, when the caller knows them (a probed count): it is authoritative over `duration × fps`, which
   * container rounding can make disagree. Without it the count is derived from the duration. Needs `fps`.
   */
  frameCount?: number;
  /** Controlled loop state; pair with `onLoopChange`. */
  loop?: boolean;
  /** Initial loop state when uncontrolled. */
  defaultLoop?: boolean;
  onLoopChange?: (loop: boolean) => void;
  /**
   * While looping, play this window instead of the whole clip. It is clamped to the media; a collapsed or inverted
   * range is no window, so the whole clip loops. Seeking outside it keeps it: retiring it is the caller's call
   * (`onSeeking`).
   */
  loopRange?: LoopRange | null;
  /** `compact` fits a small box: play, loop, scrubber and one readout; volume and fullscreen are left out. */
  size?: 'compact' | 'full';
  /** Caller content above the video and below the controls; the caller positions it. */
  overlay?: ReactNode;
  /** The video element, for behavior the player does not own. */
  ref?: Ref<HTMLVideoElement>;
}

type PlayerSize = NonNullable<VideoPlayerProps['size']>;

/** The frames of the clip; null without a frame rate. */
interface FrameTiming {
  fps: number;
  /** Authoritative count from the caller, or null to derive it from the duration. */
  frameCount: number | null;
}

const resolveFrameCount = ({ fps, frameCount }: FrameTiming, duration: number): number =>
  frameCount ?? getFrameCount(duration, fps);

const PLAYBACK_RATES = [0.25, 0.5, 1, 2] as const;
const CONTROLS_IDLE_MS = 2000;
const VIDEO_STYLE: CSSProperties = { display: 'block', height: '100%', objectFit: 'contain', width: '100%' };
const CONTROLS_ATTRIBUTE = 'data-video-player-controls';

// Idle controls fade out, but stay focusable: keyboard focus inside the player, or the pointer resting on the bar,
// keeps them shown.
const CONTROLS_SHOWN_ANYWAY = `:has(:focus-visible, [${CONTROLS_ATTRIBUTE}]:hover)`;
const ROOT_CSS = {
  [`&[data-controls-hidden]:not(${CONTROLS_SHOWN_ANYWAY})`]: { cursor: 'none' },
  [`&[data-controls-hidden]:not(${CONTROLS_SHOWN_ANYWAY}) [${CONTROLS_ATTRIBUTE}]`]: {
    opacity: 0,
    pointerEvents: 'none',
  },
};

// The controls sit on a dark scrim in every theme, so they use fixed light colors.
const SCRIM_BUTTON_HOVER = { bg: 'whiteAlpha.200' } as const;
// The loop window is shaded under the progress fill (`::before` paints before the absolutely placed range).
const SCRIM_SLIDER_CSS = {
  '--slider-track-size': 'sizes.1',
  '& [data-part="track"]': { bg: 'whiteAlpha.300', position: 'relative' },
  '&[data-loop-window] [data-part="track"]::before': {
    bg: 'whiteAlpha.500',
    content: '""',
    insetBlock: '0',
    insetInlineEnd: 'var(--video-loop-end)',
    insetInlineStart: 'var(--video-loop-start)',
    pointerEvents: 'none',
    position: 'absolute',
  },
};
const MOTION_REDUCE = { transitionDuration: '0s' } as const;
const CONTROLS_BAR_PROPS = {
  [CONTROLS_ATTRIBUTE]: '',
  _motionReduce: MOTION_REDUCE,
  bgGradient: 'to-t',
  bottom: '0',
  color: 'white',
  gradientFrom: 'blackAlpha.800',
  gradientTo: 'transparent',
  insetX: '0',
  position: 'absolute',
  transitionDuration: 'moderate',
  transitionProperty: 'opacity',
} as const;
const RATE_MENU_POSITIONING = { placement: 'top-end' } as const;

/**
 * The player's shortcuts. While focus is inside the player they are always its own, even when one does nothing in
 * the current state or size, so they never reach application hotkeys behind it.
 */
const PLAYER_KEYS = new Set([' ', ',', '.', 'f', 'k', 'l', 'm']);
// Keys these targets own outright, and the activation key buttons and links own.
const KEY_OWNER_SELECTOR = 'input, textarea, select, [contenteditable]:not([contenteditable="false"]), [role="menu"]';
const ACTIVATION_OWNER_SELECTOR = 'button, a[href], [role="button"]';

interface PlayerStatus {
  duration: number;
  error: boolean;
  paused: boolean;
  ready: boolean;
  waiting: boolean;
}

const selectStatus = (snapshot: MediaElementSnapshot): PlayerStatus => ({
  duration: snapshot.duration,
  error: snapshot.error,
  paused: snapshot.paused,
  ready: snapshot.ready,
  waiting: snapshot.waiting,
});
const selectPlaybackRate = (snapshot: MediaElementSnapshot): number => snapshot.playbackRate;
const selectVolume = (snapshot: MediaElementSnapshot) => ({ muted: snapshot.muted, volume: snapshot.volume });

const useMediaSelector = <Selected,>(
  store: MediaElementStore,
  selector: (snapshot: MediaElementSnapshot) => Selected
): Selected => useExternalStoreSelector(store.subscribe, store.getSnapshot, selector);

const subscribeFullscreenChange = (listener: () => void): (() => void) => {
  document.addEventListener('fullscreenchange', listener);

  return () => document.removeEventListener('fullscreenchange', listener);
};
const getFullscreenElement = (): Element | null => document.fullscreenElement;
const getNoFullscreenElement = (): Element | null => null;

/** Registers the loop window for one mount; key it by the window so a new window remounts it. */
const LoopWindowBinding = ({ range, store }: { range: LoopRange; store: MediaElementStore }) => {
  useMountEffect(() => store.setLoopWindow(range));

  return null;
};

const ShortcutLabel = ({ label, shortcut }: { label: string; shortcut: string }) => (
  <HStack gap="1.5">
    <Text>{label}</Text>
    <Kbd variant="subtle">{shortcut}</Kbd>
  </HStack>
);

interface ControlButtonProps {
  disabled: boolean;
  icon: LucideIcon;
  label: string;
  /** Inside a fullscreen player, popups render in place: only the fullscreen subtree is visible. */
  portalled: boolean;
  pressed?: boolean;
  shortcut: string;
  size: PlayerSize;
  onClick: () => void;
}

const ControlButton = ({ disabled, icon, label, onClick, portalled, pressed, shortcut, size }: ControlButtonProps) => {
  const content = useMemo(() => <ShortcutLabel label={label} shortcut={shortcut} />, [label, shortcut]);

  return (
    <Tooltip content={content} portalled={portalled}>
      <IconButton
        _hover={pressed ? undefined : SCRIM_BUTTON_HOVER}
        aria-label={label}
        aria-pressed={pressed}
        color={pressed ? undefined : 'white'}
        disabled={disabled}
        flexShrink={0}
        size={size === 'compact' ? 'xs' : 'sm'}
        variant={pressed ? 'solid' : 'ghost'}
        onClick={onClick}
      >
        <Icon as={icon} boxSize={size === 'compact' ? '3' : '3.5'} />
      </IconButton>
    </Tooltip>
  );
};

const ARROW_DIRECTIONS: Partial<Record<string, number>> = { ArrowDown: -1, ArrowLeft: -1, ArrowRight: 1, ArrowUp: 1 };

const Scrubber = ({
  disabled,
  frames,
  loopWindow,
  onScrubEnd,
  onScrubStart,
  store,
}: {
  disabled: boolean;
  frames: FrameTiming | null;
  loopWindow: LoopRange | null;
  onScrubEnd: () => void;
  onScrubStart: () => void;
  store: MediaElementStore;
}) => {
  const { t } = useTranslation();
  // Quantized to the slider's step, so frames that cannot move the thumb do not re-render it.
  const selectPosition = useCallback(
    ({ currentTime, duration }: MediaElementSnapshot) => {
      if (frames === null) {
        const { pointer } = getScrubSteps(duration);

        return { duration, frameCount: 0, position: Math.min(duration, Math.round(currentTime / pointer) * pointer) };
      }

      const frameCount = resolveFrameCount(frames, duration);

      // The thumb sits on its frame's start, so a keyboard step lands on the next frame, not past it.
      return { duration, frameCount, position: getFrameAtTime(currentTime, frames.fps, frameCount) / frames.fps };
    },
    [frames]
  );
  const { duration, frameCount, position } = useMediaSelector(store, selectPosition);
  // With frames the last position is the last frame's start, so the thumb reaches the end on the last frame.
  const max = frames === null ? duration : (frameCount - 1) / frames.fps;
  const isScrubbable = max > 0;
  const steps = frames === null ? getScrubSteps(duration) : { keyboard: 1 / frames.fps, pointer: 1 / frames.fps };
  const value = useMemo(() => [position], [position]);
  const loopStyle = useMemo(() => {
    if (!loopWindow || !isScrubbable) {
      return undefined;
    }

    const percentOf = (seconds: number) => Math.min(100, Math.max(0, (seconds / max) * 100));

    return {
      '--video-loop-end': `${100 - percentOf(loopWindow.endSeconds)}%`,
      '--video-loop-start': `${percentOf(loopWindow.startSeconds)}%`,
    } as CSSProperties;
  }, [isScrubbable, loopWindow, max]);
  const ariaLabel = useMemo(() => [t('common.videoPlayer.seek')], [t]);
  const frameOf = useCallback(
    (seconds: number) => (frames === null ? 0 : getFrameAtTime(seconds, frames.fps, frameCount)),
    [frameCount, frames]
  );
  const formatValue = useCallback(
    (seconds: number) =>
      frames === null ? formatMediaTime(seconds) : t('common.videoPlayer.frame', { frame: frameOf(seconds) }),
    [frameOf, frames, t]
  );
  const getAriaValueText = useCallback(
    ({ value: seconds }: { value: number }) =>
      frames === null
        ? t('common.videoPlayer.timeOfDuration', {
            current: formatMediaTime(seconds),
            duration: formatMediaTime(duration),
          })
        : t('common.videoPlayer.frameOfLast', { frame: frameOf(seconds), last: frameCount - 1 }),
    [duration, frameCount, frameOf, frames, t]
  );
  // Capture phase: the slider's control stops the pointerdown from bubbling. The conditions are the slider's own for
  // starting a drag (primary button, no Ctrl/Alt/Meta), so every scrub this starts is one the slider ends or cancels.
  const handlePointerDownCapture = useCallback(
    (event: PointerEvent<HTMLDivElement>) => {
      if (disabled || !isScrubbable || event.button !== 0 || event.ctrlKey || event.altKey || event.metaKey) {
        return;
      }

      onScrubStart();
    },
    [disabled, isScrubbable, onScrubStart]
  );
  // The slider's step is fine so a drag lands anywhere; arrow keys move by the coarser keyboard step instead, which
  // the slider cannot take separately. A handled key never reaches the slider (it ignores prevented keys).
  const handleKeyDownCapture = useCallback(
    (event: KeyboardEvent<HTMLDivElement>) => {
      const direction = ARROW_DIRECTIONS[event.key];

      if (
        frames !== null ||
        direction === undefined ||
        disabled ||
        !isScrubbable ||
        event.altKey ||
        event.ctrlKey ||
        event.metaKey
      ) {
        return;
      }

      event.preventDefault();
      event.stopPropagation();
      const step = event.shiftKey ? steps.keyboard * 10 : steps.keyboard;
      store.seek(store.getSnapshot().currentTime + direction * step);
    },
    [disabled, frames, isScrubbable, steps.keyboard, store]
  );
  const handleValueChange = useCallback(
    ({ value: [seconds] }: { value: number[] }) => {
      if (seconds !== undefined) {
        store.seek(frames === null ? seconds : getFrameTime(frameOf(seconds), frames.fps));
      }
    },
    [frameOf, frames, store]
  );

  return (
    <Slider
      aria-label={ariaLabel}
      colorPalette="accent"
      css={SCRIM_SLIDER_CSS}
      data-loop-window={loopStyle ? '' : undefined}
      disabled={disabled || !isScrubbable}
      flex="1"
      formatValue={formatValue}
      getAriaValueText={getAriaValueText}
      largeStep={steps.keyboard * 10}
      max={isScrubbable ? max : 1}
      min={0}
      minW="0"
      size="md"
      step={steps.pointer}
      style={loopStyle}
      thumbAlignment="center"
      value={value}
      onKeyDownCapture={handleKeyDownCapture}
      onPointerDownCapture={handlePointerDownCapture}
      onValueChange={handleValueChange}
      onValueChangeEnd={onScrubEnd}
    />
  );
};

const READOUT_PROPS = {
  color: 'whiteAlpha.900',
  flexShrink: 0,
  fontVariantNumeric: 'tabular-nums',
  whiteSpace: 'nowrap',
} as const;

const TimeReadout = ({
  frames,
  size,
  store,
}: {
  frames: FrameTiming | null;
  size: PlayerSize;
  store: MediaElementStore;
}) => {
  const { t } = useTranslation();
  const selectReadout = useCallback(
    ({ currentTime, duration }: MediaElementSnapshot) => {
      if (frames === null) {
        return {
          frame: null,
          time:
            size === 'compact'
              ? formatMediaTime(currentTime)
              : `${formatMediaTime(currentTime)} / ${formatMediaTime(duration)}`,
        };
      }

      const frameCount = resolveFrameCount(frames, duration);

      return {
        frame: `${getFrameAtTime(currentTime, frames.fps, frameCount)} / ${frameCount - 1}`,
        time: `${formatMediaTime(currentTime)} / ${formatMediaTime(duration)}`,
      };
    },
    [frames, size]
  );
  const { frame, time } = useMediaSelector(store, selectReadout);

  if (size === 'compact') {
    return (
      <Text {...READOUT_PROPS} fontSize="xs">
        {frame ?? time}
      </Text>
    );
  }

  return (
    <HStack {...READOUT_PROPS} fontSize="sm" gap="2">
      <Text>{time}</Text>
      {frame === null ? null : <Text color="whiteAlpha.800">{t('common.videoPlayer.frame', { frame })}</Text>}
    </HStack>
  );
};

const PlaybackRateMenu = ({
  disabled,
  onOpenChange,
  portalled,
  store,
}: {
  disabled: boolean;
  onOpenChange: (open: boolean) => void;
  portalled: boolean;
  store: MediaElementStore;
}) => {
  const { t } = useTranslation();
  const ids = useTooltipTriggerIds();
  const rate = useMediaSelector(store, selectPlaybackRate);
  const rateLabel = t('common.videoPlayer.rate', { rate });
  const handleOpenChange = useCallback(({ open }: { open: boolean }) => onOpenChange(open), [onOpenChange]);
  const handleValueChange = useCallback(
    ({ value }: { value: string }) => store.setPlaybackRate(Number(value)),
    [store]
  );

  return (
    <Menu.Root ids={ids} positioning={RATE_MENU_POSITIONING} onOpenChange={handleOpenChange}>
      <Tooltip content={t('common.videoPlayer.playbackSpeed')} ids={ids} portalled={portalled}>
        <Menu.Trigger asChild>
          <Button
            _hover={SCRIM_BUTTON_HOVER}
            aria-label={t('common.videoPlayer.playbackSpeedValue', { rate: rateLabel })}
            color="white"
            disabled={disabled}
            flexShrink={0}
            fontVariantNumeric="tabular-nums"
            minW="9"
            size="sm"
            variant="ghost"
          >
            {rateLabel}
          </Button>
        </Menu.Trigger>
      </Tooltip>
      <Portal disabled={!portalled}>
        <Menu.Positioner>
          <MenuContent minW="6rem">
            <Menu.RadioItemGroup value={String(rate)} onValueChange={handleValueChange}>
              {PLAYBACK_RATES.map((option) => (
                <Menu.RadioItem key={option} value={String(option)}>
                  <Menu.ItemIndicator />
                  <Menu.ItemText>{t('common.videoPlayer.rate', { rate: option })}</Menu.ItemText>
                </Menu.RadioItem>
              ))}
            </Menu.RadioItemGroup>
          </MenuContent>
        </Menu.Positioner>
      </Portal>
    </Menu.Root>
  );
};

const formatVolume = (volume: number): string => `${Math.round(volume * 100)}%`;
const getVolumeValueText = ({ value }: { value: number }): string => formatVolume(value);

const VolumeControls = ({
  disabled,
  onToggleMute,
  portalled,
  store,
}: {
  disabled: boolean;
  onToggleMute: () => void;
  portalled: boolean;
  store: MediaElementStore;
}) => {
  const { t } = useTranslation();
  const { muted, volume } = useMediaSelector(store, selectVolume);
  const silent = muted || volume === 0;
  const value = useMemo(() => [silent ? 0 : volume], [silent, volume]);
  const ariaLabel = useMemo(() => [t('common.videoPlayer.volume')], [t]);
  const handleValueChange = useCallback(
    ({ value: [next] }: { value: number[] }) => {
      if (next !== undefined) {
        store.setVolume(next);
      }
    },
    [store]
  );

  return (
    <HStack flexShrink={0} gap="1.5">
      <ControlButton
        disabled={disabled}
        icon={silent ? VolumeXIcon : volume < 0.5 ? Volume1Icon : Volume2Icon}
        label={silent ? t('common.videoPlayer.unmute') : t('common.videoPlayer.mute')}
        portalled={portalled}
        shortcut="M"
        size="full"
        onClick={onToggleMute}
      />
      <Slider
        aria-label={ariaLabel}
        colorPalette="accent"
        css={SCRIM_SLIDER_CSS}
        disabled={disabled}
        formatValue={formatVolume}
        getAriaValueText={getVolumeValueText}
        max={1}
        min={0}
        size="md"
        step={0.05}
        value={value}
        w="16"
        onValueChange={handleValueChange}
      />
    </HStack>
  );
};

/**
 * A video with its own controls: play, loop (on by default, optionally over a window), scrubbing, frame steps with
 * `fps`, speed, volume and fullscreen. Shortcuts act only while focus is inside the player. Hiding or unmounting the
 * player pauses it, so audio never outlives its controls. Media errors leave the picture to the caller (`onError` and
 * `overlay`) and remove the controls, since none of them could work.
 */
export const VideoPlayer = ({
  'aria-label': label,
  defaultLoop = true,
  fps: fpsProp,
  frameCount: frameCountProp,
  loop: loopProp,
  loopRange,
  onClick,
  onEmptied,
  onError,
  onLoopChange,
  onPlay,
  overlay,
  ref,
  size = 'full',
  style,
  ...videoProps
}: VideoPlayerProps) => {
  const { t } = useTranslation();
  const [store] = useState(createMediaElementStore);
  const status = useMediaSelector(store, selectStatus);
  const fps = fpsProp !== undefined && Number.isFinite(fpsProp) && fpsProp > 0 ? fpsProp : null;
  const frameCount =
    frameCountProp !== undefined && Number.isInteger(frameCountProp) && frameCountProp > 0 ? frameCountProp : null;
  const frames = useMemo(() => (fps === null ? null : { fps, frameCount }), [fps, frameCount]);
  const [uncontrolledLoop, setUncontrolledLoop] = useState(defaultLoop);
  const loop = loopProp ?? uncontrolledLoop;
  const loopWindow = loop && loopRange ? clampLoopRange(loopRange, status.duration) : null;
  const rootId = useId();
  const rootRef = useRef<HTMLDivElement>(null);
  const videoElementRef = useRef<HTMLVideoElement>(null);
  const fullscreenElement = useSyncExternalStore(
    subscribeFullscreenChange,
    getFullscreenElement,
    getNoFullscreenElement
  );
  const isFullscreen = fullscreenElement !== null && fullscreenElement.id === rootId;
  const canFullscreen = size === 'full' && document.fullscreenEnabled;
  const [isIdle, setIsIdle] = useState(false);
  // Set while a pointer drag on the scrubber holds playback paused; `resume` is whether it was playing before.
  const [scrub, setScrub] = useState<{ resume: boolean } | null>(null);
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const idleTimerRef = useRef<number | null>(null);
  // A mid-playback error can leave metadata in place, but nothing the controls do would work.
  const disabled = !status.ready || status.error;

  // The slider cancels a drag without ending it when it becomes disabled (a new source, an error); drop the scrub
  // rather than leave playback paused by it and the controls pinned. Adjusted during render, as derived state.
  if (scrub !== null && disabled) {
    setScrub(null);
  }

  const controlsHidden = !status.paused && isIdle && scrub === null && !isMenuOpen;
  const portalled = !isFullscreen;

  const attachVideo = useCallback(
    (video: HTMLVideoElement | null) => (video ? store.attach(video) : undefined),
    [store]
  );
  const videoRef = useMemo(() => mergeRefs(attachVideo, videoElementRef, ref), [attachVideo, ref]);
  const videoStyle = useMemo(() => (style ? { ...VIDEO_STYLE, ...style } : VIDEO_STYLE), [style]);

  const clearIdleTimer = useCallback(() => {
    if (idleTimerRef.current !== null) {
      window.clearTimeout(idleTimerRef.current);
      idleTimerRef.current = null;
    }
  }, []);
  const wake = useCallback(() => {
    clearIdleTimer();
    setIsIdle(false);
    idleTimerRef.current = window.setTimeout(() => {
      idleTimerRef.current = null;
      setIsIdle(true);
    }, CONTROLS_IDLE_MS);
  }, [clearIdleTimer]);
  // Hiding (Activity) or unmounting must silence the clip. Here rather than in the ref cleanup, which also runs when a
  // caller's ref changes identity; a hidden Activity detaches refs first, hence the element captured at mount.
  useMountEffect(() => {
    const video = videoElementRef.current;

    return () => {
      clearIdleTimer();

      if (video && !video.paused) {
        video.pause();
      }
    };
  });

  const setLoop = useCallback(
    (next: boolean) => {
      if (loopProp === undefined) {
        setUncontrolledLoop(next);
      }

      onLoopChange?.(next);
    },
    [loopProp, onLoopChange]
  );
  const toggleLoop = useCallback(() => setLoop(!loop), [loop, setLoop]);
  // Pause while dragging so playback does not fight the thumb; resume once the drag ends. A second pointer joining a
  // drag keeps the first one's resume.
  const startScrub = useCallback(() => {
    const resume = !store.getSnapshot().paused;

    setScrub((current) => current ?? { resume });
    store.pause();
  }, [store]);
  // Also called for keyboard changes, which start no scrub.
  const endScrub = useCallback(() => {
    if (scrub === null) {
      return;
    }

    setScrub(null);

    if (scrub.resume) {
      store.play();
    }
  }, [scrub, store]);
  const toggleMute = useCallback(() => {
    const { muted, volume } = store.getSnapshot();

    if (muted || volume === 0) {
      store.setVolume(volume === 0 ? 1 : volume);
    } else {
      store.setMuted(true);
    }
  }, [store]);
  const stepFrame = useCallback(
    (delta: number) => {
      if (frames === null) {
        return;
      }

      const { currentTime, duration } = store.getSnapshot();
      const count = resolveFrameCount(frames, duration);
      const frame = Math.min(count - 1, Math.max(0, getFrameAtTime(currentTime, frames.fps, count) + delta));

      store.pause();
      store.seek(getFrameTime(frame, frames.fps));
    },
    [frames, store]
  );
  const stepBackward = useCallback(() => stepFrame(-1), [stepFrame]);
  const stepForward = useCallback(() => stepFrame(1), [stepFrame]);
  const toggleFullscreen = useCallback(() => {
    const root = rootRef.current;

    if (!root) {
      return;
    }

    if (document.fullscreenElement === root) {
      void document.exitFullscreen().catch(() => {});
    } else {
      void root.requestFullscreen().catch(() => {});
    }
  }, []);

  const handleKeyDown = useCallback(
    (event: KeyboardEvent<HTMLDivElement>) => {
      const target = event.target;
      const key = event.key.toLowerCase();

      if (
        !PLAYER_KEYS.has(key) ||
        event.defaultPrevented ||
        event.altKey ||
        event.ctrlKey ||
        event.metaKey ||
        event.shiftKey ||
        isImeComposing(event.nativeEvent) ||
        // React delivers keys from portalled popups here too; those belong to the popup.
        !(target instanceof Element) ||
        !event.currentTarget.contains(target) ||
        target.closest(KEY_OWNER_SELECTOR) ||
        (key === ' ' && target.closest(ACTIVATION_OWNER_SELECTOR))
      ) {
        return;
      }

      event.preventDefault();
      event.stopPropagation();

      if (disabled) {
        return;
      }

      switch (key) {
        case ' ':
        case 'k':
          store.togglePlay();
          break;
        case 'l':
          toggleLoop();
          break;
        case ',':
          stepFrame(-1);
          break;
        case '.':
          stepFrame(1);
          break;
        case 'm':
          if (size === 'full') {
            toggleMute();
          }
          break;
        case 'f':
          if (canFullscreen) {
            toggleFullscreen();
          }
          break;
      }

      wake();
    },
    [canFullscreen, disabled, size, stepFrame, store, toggleFullscreen, toggleLoop, toggleMute, wake]
  );
  const handlePointerLeave = useCallback(
    (event: PointerEvent<HTMLDivElement>) => {
      // Touch pointers leave on every lift; only a mouse leaving means nobody is looking at the controls.
      if (event.pointerType === 'mouse') {
        clearIdleTimer();
        setIsIdle(true);
      }
    },
    [clearIdleTimer]
  );
  const handleVideoClick = useCallback(
    (event: MouseEvent<HTMLVideoElement>) => {
      onClick?.(event);

      // A double-click's second click is not a second toggle.
      if (!event.defaultPrevented && !disabled && event.detail <= 1) {
        store.togglePlay();
      }
    },
    [disabled, onClick, store]
  );
  const handlePlay = useCallback(
    (event: SyntheticEvent<HTMLVideoElement>) => {
      onPlay?.(event);
      wake();
    },
    [onPlay, wake]
  );
  // A new source disables the controls and an error removes them; either would drop a focused control's focus to the
  // page, so it moves to the player first.
  const keepFocusInPlayer = useCallback(() => {
    const root = rootRef.current;

    if (root && document.activeElement !== root && root.contains(document.activeElement)) {
      root.focus();
    }
  }, []);
  const handleEmptied = useCallback(
    (event: SyntheticEvent<HTMLVideoElement>) => {
      keepFocusInPlayer();
      onEmptied?.(event);
    },
    [keepFocusInPlayer, onEmptied]
  );
  const handleError = useCallback(
    (event: SyntheticEvent<HTMLVideoElement>) => {
      keepFocusInPlayer();
      onError?.(event);
    },
    [keepFocusInPlayer, onError]
  );

  const loopButton = (
    <ControlButton
      disabled={disabled}
      icon={RepeatIcon}
      label={t('common.videoPlayer.loop')}
      portalled={portalled}
      pressed={loop}
      shortcut="L"
      size={size}
      onClick={toggleLoop}
    />
  );
  const playButton = (
    <ControlButton
      disabled={disabled}
      icon={status.paused ? PlayIcon : PauseIcon}
      label={status.paused ? t('common.videoPlayer.play') : t('common.videoPlayer.pause')}
      portalled={portalled}
      shortcut="K"
      size={size}
      onClick={store.togglePlay}
    />
  );
  const scrubber = (
    <Scrubber
      disabled={disabled}
      frames={frames}
      loopWindow={loopWindow}
      store={store}
      onScrubEnd={endScrub}
      onScrubStart={startScrub}
    />
  );

  return (
    <Box
      ref={rootRef}
      aria-label={label}
      bg="black"
      css={ROOT_CSS}
      data-controls-hidden={controlsHidden ? '' : undefined}
      h="full"
      id={rootId}
      overflow="hidden"
      position="relative"
      role="group"
      tabIndex={-1}
      w="full"
      onKeyDown={handleKeyDown}
      onPointerDown={wake}
      onPointerLeave={handlePointerLeave}
      onPointerMove={wake}
    >
      {/* Callers play arbitrary user media, which carries no caption-track contract. */}
      {/* oxlint-disable-next-line jsx-a11y/media-has-caption */}
      <video
        ref={videoRef}
        playsInline
        preload="metadata"
        {...videoProps}
        loop={loop && loopWindow === null}
        style={videoStyle}
        onClick={handleVideoClick}
        onEmptied={handleEmptied}
        onError={handleError}
        onPlay={handlePlay}
      />
      {loopWindow ? (
        <LoopWindowBinding
          key={`${loopWindow.startSeconds}:${loopWindow.endSeconds}`}
          range={loopWindow}
          store={store}
        />
      ) : null}
      {overlay}
      {status.waiting ? (
        <Center inset="0" pointerEvents="none" position="absolute">
          <Spinner color="white" size={size === 'compact' ? 'md' : 'lg'} />
        </Center>
      ) : null}
      {/* Mounted throughout so the region exists before it announces. */}
      <chakra.span aria-live="polite" srOnly>
        {status.waiting ? t('common.videoPlayer.buffering') : null}
      </chakra.span>
      {status.error ? null : size === 'compact' ? (
        <HStack {...CONTROLS_BAR_PROPS} gap="1" pb="1" pt="4" px="1.5">
          {playButton}
          {loopButton}
          {/* The thumb centers on the track ends, so the track needs half a thumb of room each side. */}
          <HStack flex="1" minW="0" px="1.5">
            {scrubber}
          </HStack>
          <TimeReadout frames={frames} size="compact" store={store} />
        </HStack>
      ) : (
        <Stack {...CONTROLS_BAR_PROPS} gap="0.5" pb="1.5" pt="8" px="2">
          <HStack px="1.5">{scrubber}</HStack>
          <HStack gap="1" minW="0">
            {playButton}
            {frames === null ? null : (
              <>
                <ControlButton
                  disabled={disabled}
                  icon={StepBackIcon}
                  label={t('common.videoPlayer.previousFrame')}
                  portalled={portalled}
                  shortcut=","
                  size="full"
                  onClick={stepBackward}
                />
                <ControlButton
                  disabled={disabled}
                  icon={StepForwardIcon}
                  label={t('common.videoPlayer.nextFrame')}
                  portalled={portalled}
                  shortcut="."
                  size="full"
                  onClick={stepForward}
                />
              </>
            )}
            {loopButton}
            <TimeReadout frames={frames} size="full" store={store} />
            <Box flex="1" />
            <PlaybackRateMenu disabled={disabled} portalled={portalled} store={store} onOpenChange={setIsMenuOpen} />
            <VolumeControls disabled={disabled} portalled={portalled} store={store} onToggleMute={toggleMute} />
            {canFullscreen ? (
              <ControlButton
                disabled={disabled}
                icon={isFullscreen ? MinimizeIcon : MaximizeIcon}
                label={isFullscreen ? t('common.videoPlayer.exitFullscreen') : t('common.videoPlayer.fullscreen')}
                portalled={portalled}
                shortcut="F"
                size="full"
                onClick={toggleFullscreen}
              />
            ) : null}
          </HStack>
        </Stack>
      )}
    </Box>
  );
};
