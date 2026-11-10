import { HStack, Text } from '@chakra-ui/react';
import { useMountEffect } from '@platform/react/useMountEffect';
import { IconButton } from '@platform/ui/Button';
import { ChevronDownIcon, ChevronUpIcon } from 'lucide-react';
import { type KeyboardEvent as ReactKeyboardEvent, useCallback, useRef } from 'react';
import { useTranslation } from 'react-i18next';

/**
 * Preview mode's stepper: it steps the workflow the detail rail describes through the view's current results, so it
 * sits with that workflow's name (atop the rail, or beside the name in the narrow sheet's summary). Up and Down on it
 * step too.
 */

export interface WorkflowPreviewStepperProps {
  /** The workflow's place in the view's current results (0-based); null while unknown. */
  position: number | null;
  total: number;
  canStepPrevious: boolean;
  canStepNext: boolean;
  /**
   * Steps through the results. A step that must first read its position asks `isCurrent` before selecting what
   * arrived: false once the stepper has gone (the preview closed) or a later step has replaced it.
   */
  onStep: (delta: -1 | 1, isCurrent: () => boolean) => void;
}

export const WorkflowPreviewStepper = ({
  canStepNext,
  canStepPrevious,
  position,
  total,
  onStep,
}: WorkflowPreviewStepperProps) => {
  const { t } = useTranslation();
  // The latest step, and whether this stepper still shows: a read that lands after either changed selects nothing.
  const stepToken = useRef(0);
  const isMounted = useRef(false);
  useMountEffect(() => {
    isMounted.current = true;

    return () => {
      isMounted.current = false;
    };
  });
  const step = useCallback(
    (delta: -1 | 1) => {
      const token = ++stepToken.current;

      onStep(delta, () => isMounted.current && stepToken.current === token);
    },
    [onStep]
  );
  const stepPrevious = useCallback(() => step(-1), [step]);
  const stepNext = useCallback(() => step(1), [step]);
  const handleKeyDown = useCallback(
    (event: ReactKeyboardEvent<HTMLDivElement>) => {
      if (event.key === 'ArrowUp' || event.key === 'ArrowDown') {
        event.preventDefault();
        step(event.key === 'ArrowUp' ? -1 : 1);
      }
    },
    [step]
  );
  // Stepping needs the workflow's place in the results; at an end there is nowhere further to go.
  const unavailable = (canStep: boolean, endKey: string) =>
    position === null ? t('workflowLibrary.preview.positionUnknown') : canStep ? undefined : t(endKey);

  return (
    <HStack
      aria-label={t('workflowLibrary.preview.stepper')}
      data-preview-stepper=""
      flexShrink={0}
      gap="1"
      justify="space-between"
      minW="0"
      role="group"
      onKeyDown={handleKeyDown}
    >
      <Text
        aria-live="polite"
        color="fg.muted"
        data-preview-position=""
        fontSize="md"
        fontVariantNumeric="tabular-nums"
        whiteSpace="nowrap"
      >
        {position !== null ? t('workflowLibrary.preview.position', { position: position + 1, total }) : null}
      </Text>
      <HStack gap="0.5">
        <IconButton
          aria-label={t('workflowLibrary.preview.previous')}
          size="sm"
          tooltip={t('workflowLibrary.preview.previous')}
          unavailableReason={unavailable(canStepPrevious, 'workflowLibrary.preview.atFirst')}
          variant="ghost"
          onClick={stepPrevious}
        >
          <ChevronUpIcon />
        </IconButton>
        <IconButton
          aria-label={t('workflowLibrary.preview.next')}
          size="sm"
          tooltip={t('workflowLibrary.preview.next')}
          unavailableReason={unavailable(canStepNext, 'workflowLibrary.preview.atLast')}
          variant="ghost"
          onClick={stepNext}
        >
          <ChevronDownIcon />
        </IconButton>
      </HStack>
    </HStack>
  );
};
