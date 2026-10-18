/* eslint-disable react-perf/jsx-no-new-function-as-prop */
import type { WorkingMemoryPlan } from '@features/models/core/workingMemory';

import { chakra, Portal, Stack, Text } from '@chakra-ui/react';
import { planCalibrationSteps } from '@features/models/core/workingMemory';
import { Button, CloseButton } from '@platform/ui/Button';
import { Dialog } from '@platform/ui/Dialog';
import { useCallback, useRef } from 'react';
import { useTranslation } from 'react-i18next';

import { SharedModelsList, useCaseLabel } from './WorkingMemoryParts';

const SECONDS_PER_MINUTE = 60;

/** Confirms a calibration: what it runs, where, for how long, and which other models it moves. */
export const WorkingMemoryCalibrateDialog = ({
  calibratedThroughName,
  deviceName,
  getFinalFocus,
  isOpen,
  onClose,
  onConfirm,
  onExitComplete,
  plan,
}: {
  /** The main model a standalone VAE's cases generate with. */
  calibratedThroughName: string | null;
  deviceName: string | null;
  /** Where focus goes once the dialog closes; starting a calibration can disable the opener. */
  getFinalFocus: () => HTMLElement | null;
  isOpen: boolean;
  onClose: () => void;
  onConfirm: () => void;
  onExitComplete: () => void;
  plan: WorkingMemoryPlan;
}) => {
  const { t } = useTranslation();
  const startRef = useRef<HTMLButtonElement>(null);
  const getInitialFocus = useCallback(() => startRef.current, []);
  const runCount = planCalibrationSteps(plan).length;
  const minutes =
    plan.estimated_seconds === null ? null : Math.max(1, Math.round(plan.estimated_seconds / SECONDS_PER_MINUTE));
  const caseLabel = useCaseLabel();

  return (
    <Dialog.Root
      finalFocusEl={getFinalFocus}
      initialFocusEl={getInitialFocus}
      open={isOpen}
      size="md"
      onExitComplete={onExitComplete}
      onOpenChange={(event) => {
        if (!event.open) {
          onClose();
        }
      }}
    >
      <Portal>
        <Dialog.Backdrop />
        <Dialog.Positioner>
          <Dialog.Content>
            <Dialog.Header borderBottomWidth="1px" borderColor="border.subtle">
              <Stack gap="0.5">
                <Dialog.Title>{t('models.workingMemory.dialogTitle')}</Dialog.Title>
                {deviceName ? (
                  <Dialog.Description>
                    {t('models.workingMemory.dialogDescription', { device: deviceName })}
                  </Dialog.Description>
                ) : null}
              </Stack>
            </Dialog.Header>
            <Dialog.Body>
              <Stack fontSize="sm" gap="3">
                <Stack gap="1">
                  {minutes === null ? null : (
                    <Text fontWeight="600">{t('models.workingMemory.dialogDuration', { minutes })}</Text>
                  )}
                  <Text color="fg.muted">{t('models.workingMemory.dialogQueueNote')}</Text>
                  {calibratedThroughName ? (
                    <Text color="fg.muted">
                      {t('models.workingMemory.dialogThrough', { model: calibratedThroughName })}
                    </Text>
                  ) : null}
                </Stack>
                <Stack gap="1">
                  <Text as="h3" fontWeight="600">
                    {t('models.workingMemory.dialogRuns', { count: runCount })}
                  </Text>
                  <chakra.ul display="flex" flexWrap="wrap" gap="1.5" listStyleType="none" m="0" p="0">
                    {plan.cases.map((item) => (
                      <chakra.li
                        key={item.id}
                        borderColor="border.subtle"
                        borderWidth="1px"
                        fontVariantNumeric="tabular-nums"
                        px="2"
                        py="0.5"
                        rounded="sm"
                      >
                        {caseLabel(item)}
                      </chakra.li>
                    ))}
                  </chakra.ul>
                  <Text color="fg.muted">
                    {t('models.workingMemory.dialogRepetitions', {
                      measured: plan.repetitions.measured,
                      warmUp: plan.repetitions.warm_up,
                    })}
                  </Text>
                </Stack>
                {plan.shared_with.length > 0 ? <SharedModelsList models={plan.shared_with} /> : null}
              </Stack>
            </Dialog.Body>
            <Dialog.Footer>
              <Button variant="ghost" onClick={onClose}>
                {t('common.cancel')}
              </Button>
              <Button ref={startRef} variant="solid" onClick={onConfirm}>
                {t('models.workingMemory.calibrate')}
              </Button>
            </Dialog.Footer>
            <Dialog.CloseTrigger asChild>
              <CloseButton aria-label={t('common.close')} />
            </Dialog.CloseTrigger>
          </Dialog.Content>
        </Dialog.Positioner>
      </Portal>
    </Dialog.Root>
  );
};
