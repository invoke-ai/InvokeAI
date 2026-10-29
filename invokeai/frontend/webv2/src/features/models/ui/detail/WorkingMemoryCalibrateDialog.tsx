/* eslint-disable react-perf/jsx-no-new-function-as-prop */
import type { WorkingMemoryPlan } from '@features/models/core/workingMemory';

import { Alert, chakra, Portal, Stack, Text } from '@chakra-ui/react';
import { planCalibrationSteps } from '@features/models/core/workingMemory';
import { Button, CloseButton } from '@platform/ui/Button';
import { Dialog } from '@platform/ui/Dialog';
import { useCallback, type RefObject } from 'react';
import { useTranslation } from 'react-i18next';

import { SharedModelsList, useCaseLabel } from './WorkingMemoryParts';

const SECONDS_PER_MINUTE = 60;

/** Confirms a calibration: what it runs, where, for how long, and which other models it moves. */
export const WorkingMemoryCalibrateDialog = ({
  calibratedThroughName,
  confirmRef,
  deviceName,
  getFinalFocus,
  isConfirming,
  isOpen,
  onClose,
  onConfirm,
  onExitComplete,
  plan,
  planChanges,
}: {
  /** The main model a standalone VAE's cases generate with. */
  calibratedThroughName: string | null;
  /** The Calibrate button, which takes focus on opening and again after the plan changed on confirmation. */
  confirmRef: RefObject<HTMLButtonElement | null>;
  deviceName: string | null;
  /** Where focus goes once the dialog closes; starting a calibration can disable the opener. */
  getFinalFocus: () => HTMLElement | null;
  /** The plan is read again before the start; closing meanwhile takes the confirmation back. */
  isConfirming: boolean;
  isOpen: boolean;
  onClose: () => void;
  onConfirm: () => void;
  onExitComplete: () => void;
  plan: WorkingMemoryPlan;
  /** How often a confirmation found the runs changed; the dialog then shows the new plan, flagged. */
  planChanges: number;
}) => {
  const { t } = useTranslation();
  const getInitialFocus = useCallback(() => confirmRef.current, [confirmRef]);
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
                {/* Announced on insertion, keyed so each change is a new one: the click that confirmed started nothing. */}
                {planChanges > 0 ? (
                  <Alert.Root key={planChanges} role="alert" status="warning" variant="surface">
                    <Alert.Indicator />
                    <Alert.Content>
                      <Alert.Title>{t('models.workingMemory.dialogChanged')}</Alert.Title>
                    </Alert.Content>
                  </Alert.Root>
                ) : null}
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
                  <Text color="fg.muted">{t('models.workingMemory.dialogSizesNote')}</Text>
                </Stack>
                {plan.shared_with.length > 0 ? <SharedModelsList models={plan.shared_with} /> : null}
              </Stack>
            </Dialog.Body>
            <Dialog.Footer>
              <Button variant="ghost" onClick={onClose}>
                {t('common.cancel')}
              </Button>
              <Button ref={confirmRef} loading={isConfirming} variant="solid" onClick={onConfirm}>
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
