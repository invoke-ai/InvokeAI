/* oxlint-disable react-perf/jsx-no-jsx-as-prop */
import type { MouseEvent } from 'react';

import { Flex, Icon, Menu, Portal, Text } from '@chakra-ui/react';
import { Button, IconButton } from '@platform/ui/Button';
import { MenuActionItem, MenuContent } from '@platform/ui/Menu';
import { Tooltip, useTooltipTriggerIds } from '@platform/ui/Tooltip';
import { ChevronDownIcon } from 'lucide-react';
import { useCallback, useRef } from 'react';
import { useTranslation } from 'react-i18next';

const APPLY_MENU_POSITIONING = { placement: 'bottom-end' } as const;

export interface OperationSaveTarget {
  disabled: boolean;
  label: string;
  value: string;
  onSelect(): void;
}

export interface OperationRunAction {
  disabled: boolean;
  label: string;
  /** Work is in flight: the button stays its own width and label beside a spinner. */
  loading: boolean;
  onRun(): void;
}

/**
 * The verbs of a running canvas operation, one row after its form's groups: Reset at the start; Cancel, the manual run
 * and Apply (with its save-as menu) at the end. Status and errors live in the Operation section, not here.
 *
 * A retry's run button leaves the row once its run succeeds. When that run was started from the focused button and
 * focus has nowhere else to be (the button went, or went disabled while busy, and nothing took focus since), focus
 * moves to Apply, or to Cancel while Apply is disabled, so a keyboard user is not left on the page body.
 */
export const OperationActions = ({
  applying,
  canApply,
  canCancel,
  canReset,
  canSave,
  run,
  saveAsLabel,
  saveTargets,
  onApply,
  onCancel,
  onReset,
}: {
  /** The result is being applied: Apply keeps its label beside a spinner. */
  applying: boolean;
  canApply: boolean;
  canCancel: boolean;
  canReset: boolean;
  canSave: boolean;
  /** The manual run, offered while the preview does not update by itself or to retry a run that failed. */
  run: OperationRunAction | null;
  /** Names the save-as menu trigger. */
  saveAsLabel: string;
  saveTargets: readonly OperationSaveTarget[];
  onApply(): void;
  onCancel(): void;
  onReset(): void;
}) => {
  const { t } = useTranslation();
  const saveAsIds = useTooltipTriggerIds();
  const applyRef = useRef<HTMLButtonElement>(null);
  const cancelRef = useRef<HTMLButtonElement>(null);
  const runStartedFocused = useRef(false);
  const runLoading = run?.loading ?? false;
  // Re-attached when loading changes: the button going idle while it stays means the run the user started settled
  // with the button kept (it failed, or the button still offers a run), and the hand-off is dropped. Cleanup that
  // finds the button gone hands focus over a microtask later, after the same commit may have enabled Apply;
  // StrictMode's extra mount attach runs before any run.
  const runButtonRef = useCallback(
    (node: HTMLButtonElement | null) => {
      if (!node) {
        return undefined;
      }
      if (!runLoading) {
        runStartedFocused.current = false;
      }
      return () => {
        queueMicrotask(() => {
          if (node.isConnected || !runStartedFocused.current) {
            return;
          }
          runStartedFocused.current = false;
          const { activeElement, body } = node.ownerDocument;
          if (activeElement !== null && activeElement !== body) {
            return;
          }
          const target = applyRef.current && !applyRef.current.disabled ? applyRef.current : cancelRef.current;
          if (target?.isConnected && !target.disabled) {
            target.focus();
          }
        });
      };
    },
    [runLoading]
  );
  const onRun = run?.onRun;
  const onRunClick = useCallback(
    (event: MouseEvent<HTMLButtonElement>) => {
      runStartedFocused.current = event.currentTarget === event.currentTarget.ownerDocument.activeElement;
      onRun?.();
    },
    [onRun]
  );
  return (
    <Flex align="center" gap="1" minW="0" w="full">
      {/*
        Tighter padding keeps the row on one line down to a 280 px floating pane; past that, only the run button
        gives way, truncating its label, so every verb stays visible.
      */}
      <Button disabled={!canReset} flexShrink={0} px="2" variant="ghost" onClick={onReset}>
        {t('common.reset')}
      </Button>
      <Button
        ref={cancelRef}
        data-pane-action="cancel"
        disabled={!canCancel}
        flexShrink={0}
        ms="auto"
        px="2"
        variant="ghost"
        onClick={onCancel}
      >
        {t('common.cancel')}
      </Button>
      {run ? (
        <Button
          ref={runButtonRef}
          disabled={run.disabled}
          flexShrink={1}
          loading={run.loading}
          loadingText={
            <Text as="span" truncate>
              {run.label}
            </Text>
          }
          minW="0"
          px="2"
          spinnerPlacement="start"
          onClick={onRunClick}
        >
          <Text as="span" truncate>
            {run.label}
          </Text>
        </Button>
      ) : null}
      <Flex flexShrink={0}>
        <Button
          ref={applyRef}
          data-pane-action="apply"
          disabled={!canApply}
          loading={applying}
          loadingText={t('common.apply')}
          px="2"
          roundedRight="none"
          spinnerPlacement="start"
          variant="solid"
          onClick={onApply}
        >
          {t('common.apply')}
        </Button>
        <Menu.Root ids={saveAsIds} positioning={APPLY_MENU_POSITIONING}>
          <Tooltip content={saveAsLabel} ids={saveAsIds}>
            <Menu.Trigger asChild>
              <IconButton aria-label={saveAsLabel} disabled={!canSave} minW="6" roundedLeft="none" variant="solid">
                <Icon as={ChevronDownIcon} boxSize="3.5" />
              </IconButton>
            </Menu.Trigger>
          </Tooltip>
          <Portal>
            <Menu.Positioner>
              <MenuContent minW="12rem" py="1">
                {saveTargets.map((target) => (
                  <MenuActionItem
                    key={target.value}
                    disabled={target.disabled}
                    label={target.label}
                    value={target.value}
                    onSelect={target.onSelect}
                  />
                ))}
              </MenuContent>
            </Menu.Positioner>
          </Portal>
        </Menu.Root>
      </Flex>
    </Flex>
  );
};
