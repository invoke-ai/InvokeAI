import { Switch } from '@chakra-ui/react';
import { Field } from '@platform/ui';
import { ScrubberField } from '@platform/ui/ScrubberField';
import { useTranslation } from 'react-i18next';

const SWITCH_CHECKED_PROPS = { bg: 'accent.solid' };

interface FramesSlider {
  inputMax: number;
  max: number;
  min: number;
  step: number;
}

/**
 * The run's length: the Frames control, and the Auto duration switch above it when a duration head
 * is selected.
 *
 * Under auto duration Frames stays editable and becomes the ceiling the head chooses beneath -- the
 * number the run's memory is sized by -- and says so, with the range the head will actually use.
 */
export const VideoLengthControls = ({
  autoDuration,
  autoDurationActive,
  autoDurationCeiling,
  autoDurationSupported,
  durationText,
  framesSlider,
  hasDurationHead,
  numFrames,
  numFramesFromClip,
  onAutoDurationChange,
  onNumFramesChange,
}: {
  autoDuration: boolean;
  /** Auto duration is on and this mode leaves the length open. */
  autoDurationActive: boolean;
  /**
   * The longest clip the head may choose, or null when the Frames value is too short to leave it a
   * choice. Below Frames when the head's 20 s trained range is the tighter limit.
   */
  autoDurationCeiling: { frames: number; seconds: number } | null;
  autoDurationSupported: boolean;
  durationText: string | undefined;
  framesSlider: FramesSlider;
  hasDurationHead: boolean;
  numFrames: number;
  /** A conditioning clip decides the length. */
  numFramesFromClip: boolean;
  onAutoDurationChange: (details: { checked: boolean }) => void;
  onNumFramesChange: (frames: number) => void;
}) => {
  const { t } = useTranslation();

  const framesHelp = autoDurationActive
    ? autoDurationCeiling === null
      ? t('widgets.video.autoDurationTooShort')
      : t('widgets.video.autoDurationCeiling', {
          frames: autoDurationCeiling.frames,
          seconds: autoDurationCeiling.seconds.toFixed(1),
        })
    : numFramesFromClip
      ? `${t('widgets.video.framesFromClip')}${durationText ? ` ${durationText}` : ''}`
      : durationText;

  return (
    <>
      {hasDurationHead ? (
        <Field
          disabled={!autoDurationSupported}
          helpText={
            autoDurationSupported ? t('widgets.video.autoDurationHelp') : t('widgets.video.autoDurationUnavailable')
          }
          label={t('widgets.video.autoDuration')}
        >
          <Switch.Root
            checked={autoDuration && autoDurationSupported}
            disabled={!autoDurationSupported}
            onCheckedChange={onAutoDurationChange}
          >
            <Switch.HiddenInput />
            <Switch.Control _checked={SWITCH_CHECKED_PROPS}>
              <Switch.Thumb />
            </Switch.Control>
          </Switch.Root>
        </Field>
      ) : null}
      <ScrubberField
        disabled={numFramesFromClip}
        helpText={framesHelp}
        inputMax={framesSlider.inputMax}
        label={autoDurationActive ? t('widgets.video.maxFrames') : t('widgets.video.frames')}
        max={framesSlider.max}
        min={framesSlider.min}
        step={framesSlider.step}
        value={numFrames}
        onChange={onNumFramesChange}
      />
    </>
  );
};
