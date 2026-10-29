import type { CanvasSnapshotContract } from '@workbench/canvas-engine/api';

import { chakra, Flex, HStack, Icon, Stack, Text } from '@chakra-ui/react';
import { Button, IconButton } from '@platform/ui/Button';
import { ConfirmDialog } from '@platform/ui/ConfirmDialog';
import { Scrollable } from '@platform/ui/Scrollable';
import { HISTORY_MAX_ENTRIES } from '@workbench/canvas-engine/api';
import { useCanvasEngine, type CanvasEngineHandle } from '@workbench/canvas-operations/react';
import { formatRelativeTime } from '@workbench/launchpad/formatRelativeTime';
import { isCanvasInteractionLocked } from '@workbench/widgets/canvas/canvasInteractionLock';
import {
  useCanvasCanRedo,
  useCanvasCanUndo,
  useCanvasDocumentEditingLocked,
  useCanvasHistoryEpoch,
} from '@workbench/widgets/canvas/engineStoreHooks';
import { useRestorePoint } from '@workbench/widgets/canvas/useCanvasRecovery';
import { useActiveProjectSelector } from '@workbench/WorkbenchContext';
import { ArchiveRestoreIcon, ListXIcon, Redo2Icon, Undo2Icon } from 'lucide-react';
import { useCallback, useState } from 'react';
import { useTranslation } from 'react-i18next';

const STEP_HOVER_PROPS = { bg: 'bg.muted', color: 'fg' };
const EMPTY_SNAPSHOTS: readonly CanvasSnapshotContract[] = [];

/**
 * Mirror bounded engine history oldest-first, from the canvas's starting point; selecting a step replays undo/redo,
 * with future steps below the current position. Below the steps, the restore points Clear canvas and Discard all
 * keep. Steps and restores wait while an operation holds the document.
 */
export const HistoryPane = () => {
  const { t } = useTranslation();
  const engine = useCanvasEngine();

  if (!engine) {
    return (
      <Flex align="center" color="fg.muted" fontSize="md" h="full" justify="center" p="4">
        {t('widgets.properties.noCanvas')}
      </Flex>
    );
  }
  return <ConnectedHistory engine={engine} />;
};

const ConnectedHistory = ({ engine }: { engine: CanvasEngineHandle }) => {
  const { t } = useTranslation();
  useCanvasHistoryEpoch(engine);
  const canUndo = useCanvasCanUndo(engine);
  const canRedo = useCanvasCanRedo(engine);
  const editingLocked = useCanvasDocumentEditingLocked(engine);
  const { dropped = 0, evicted, future, past } = engine.history.getEntries();
  const snapshots = useActiveProjectSelector((project) => project.canvas.snapshots ?? EMPTY_SNAPSHOTS);
  const canvasLocked = useActiveProjectSelector((project) =>
    isCanvasInteractionLocked(project.canvas, project.queue.items)
  );
  const restorePoint = useRestorePoint();
  const [isClearOpen, setIsClearOpen] = useState(false);
  const lockReason = editingLocked ? t('widgets.canvas.controls.operationRunning') : undefined;
  const undo = useCallback(() => engine.history.undo(), [engine]);
  const redo = useCallback(() => engine.history.redo(), [engine]);
  const jump = useCallback((offset: number) => engine.history.stepBy(offset), [engine]);
  const openClear = useCallback(() => setIsClearOpen(true), []);
  const closeClear = useCallback(() => setIsClearOpen(false), []);
  const clear = useCallback(() => engine.history.clearHistory(), [engine]);
  const stepCount = past.length + future.length;
  const startNote =
    evicted > 0
      ? t('widgets.layers.historyPane.evicted', { count: evicted, max: HISTORY_MAX_ENTRIES })
      : dropped > 0
        ? t('widgets.layers.historyPane.dropped', { count: dropped })
        : null;

  return (
    <Flex direction="column" h="full" minH="0">
      <HStack borderBottomWidth="1px" borderColor="border.subtle" flexShrink={0} gap="1" px="2" py="1.5">
        <IconButton
          aria-label={t('widgets.canvas.commands.undo')}
          color="fg.muted"
          disabled={!canUndo}
          size="sm"
          tooltip={t('widgets.canvas.commands.undo')}
          unavailableReason={canUndo ? lockReason : undefined}
          variant="ghost"
          onClick={undo}
        >
          <Icon as={Undo2Icon} boxSize="3.5" />
        </IconButton>
        <IconButton
          aria-label={t('widgets.canvas.commands.redo')}
          color="fg.muted"
          disabled={!canRedo}
          size="sm"
          tooltip={t('widgets.canvas.commands.redo')}
          unavailableReason={canRedo ? lockReason : undefined}
          variant="ghost"
          onClick={redo}
        >
          <Icon as={Redo2Icon} boxSize="3.5" />
        </IconButton>
        <Text color="fg.muted" flex="1" fontSize="xs" fontVariantNumeric="tabular-nums">
          {t('widgets.layers.historyPane.count', { count: past.length })}
        </Text>
        <IconButton
          aria-label={t('widgets.layers.historyPane.clear')}
          color="fg.muted"
          disabled={stepCount === 0}
          size="sm"
          tooltip={t('widgets.layers.historyPane.clear')}
          unavailableReason={stepCount > 0 ? lockReason : undefined}
          variant="ghost"
          onClick={openClear}
        >
          <Icon as={ListXIcon} boxSize="3.5" />
        </IconButton>
      </HStack>
      <Scrollable flex="1" minH="0">
        <Stack aria-label={t('widgets.layers.historyPane.steps')} as="ol" gap="0" listStyleType="none" m="0" p="1">
          <HistoryStep
            isCurrent={past.length === 0}
            isFuture={false}
            isLocked={editingLocked}
            label={t('widgets.layers.historyPane.start')}
            note={startNote}
            offset={-past.length}
            onJump={jump}
          />
          {past.map((label, index) => (
            <HistoryStep
              key={`past-${index}`}
              isCurrent={index === past.length - 1}
              isFuture={false}
              isLocked={editingLocked}
              label={label}
              offset={index - (past.length - 1)}
              onJump={jump}
            />
          ))}
          {future.map((label, index) => (
            <HistoryStep
              key={`future-${index}`}
              isCurrent={false}
              isFuture
              isLocked={editingLocked}
              label={label}
              offset={index + 1}
              onJump={jump}
            />
          ))}
        </Stack>
        {stepCount === 0 ? (
          <Text color="fg.muted" fontSize="sm" px="3" py="2">
            {t('widgets.layers.historyPane.empty')}
          </Text>
        ) : null}
        {snapshots.length > 0 ? (
          <RestorePoints
            canvasLockReason={lockReason ?? (canvasLocked ? t('widgets.layers.historyPane.restoreLocked') : undefined)}
            points={snapshots}
            onRestore={restorePoint}
          />
        ) : null}
      </Scrollable>
      <ConfirmDialog
        body={t('widgets.layers.historyPane.clearConfirm', { count: stepCount })}
        confirmLabel={t('widgets.layers.historyPane.clear')}
        isOpen={isClearOpen}
        title={t('widgets.layers.historyPane.clearTitle')}
        onClose={closeClear}
        onConfirm={clear}
      />
    </Flex>
  );
};

/** Newest first: the canvas a Clear canvas or Discard all replaced, and what restoring each brings back. */
const RestorePoints = ({
  canvasLockReason,
  onRestore,
  points,
}: {
  /** Why a canvas restore must wait; a Discard all point only re-stages results, so it never waits. */
  canvasLockReason: string | undefined;
  onRestore: (point: CanvasSnapshotContract) => Promise<void>;
  points: readonly CanvasSnapshotContract[];
}) => {
  const { t } = useTranslation();
  return (
    <Stack borderColor="border.subtle" borderTopWidth="1px" gap="0.5" mt="2" px="1" py="2">
      <Text color="fg.muted" fontSize="xs" fontWeight="600" px="2" textTransform="uppercase">
        {t('widgets.layers.historyPane.restorePoints')}
      </Text>
      <Stack
        aria-label={t('widgets.layers.historyPane.restorePoints')}
        as="ul"
        gap="0"
        listStyleType="none"
        m="0"
        p="0"
      >
        {[...points].reverse().map((point) => (
          <RestorePointRow
            key={point.id}
            lockReason={point.kind === 'discard-results' ? undefined : canvasLockReason}
            point={point}
            onRestore={onRestore}
          />
        ))}
      </Stack>
    </Stack>
  );
};

const RestorePointRow = ({
  lockReason,
  onRestore,
  point,
}: {
  lockReason: string | undefined;
  onRestore: (point: CanvasSnapshotContract) => Promise<void>;
  point: CanvasSnapshotContract;
}) => {
  const { t } = useTranslation();
  const restore = useCallback(() => void onRestore(point), [onRestore, point]);
  const restores =
    point.kind === 'discard-results'
      ? t('widgets.layers.historyPane.pointResults', { count: point.stagedImages?.length ?? 0 })
      : point.stagedImages?.length
        ? t('widgets.layers.historyPane.pointCanvasAndResults', { count: point.stagedImages.length })
        : t('widgets.layers.historyPane.pointCanvas');
  return (
    <chakra.li alignItems="center" display="flex" gap="2" m="0" minH="7" px="2" py="1">
      <Stack flex="1" gap="0" minW="0">
        <Text fontSize="sm" truncate>
          {point.name}
        </Text>
        <Text color="fg.muted" fontSize="xs">
          {[formatRelativeTime(point.createdAt), restores].filter(Boolean).join(' · ')}
        </Text>
      </Stack>
      <Button size="xs" unavailableReason={lockReason} variant="outline" onClick={restore}>
        <ArchiveRestoreIcon />
        {t('widgets.layers.historyPane.restore')}
      </Button>
    </chakra.li>
  );
};

const HistoryStep = ({
  isCurrent,
  isFuture,
  isLocked,
  label,
  note,
  offset,
  onJump,
}: {
  isCurrent: boolean;
  isFuture: boolean;
  /** An operation holds the document: no step replays meanwhile. */
  isLocked: boolean;
  label: string;
  /** A second line, for the starting point's dropped steps. */
  note?: string | null;
  /** Steps to replay to make this the last applied entry; 0 for the current one. */
  offset: number;
  onJump: (offset: number) => void;
}) => {
  const jump = useCallback(() => {
    if (offset !== 0 && !isLocked) {
      onJump(offset);
    }
  }, [isLocked, offset, onJump]);

  return (
    <chakra.li display="flex" m="0" p="0">
      <chakra.button
        aria-current={isCurrent ? 'step' : undefined}
        aria-disabled={isLocked || undefined}
        bg={isCurrent ? 'bg.emphasized' : 'transparent'}
        // Future (redoable) steps read as dimmed via italics; fg.subtle fails
        // WCAG contrast at this size, so color stays at fg.muted for both.
        color={isCurrent ? 'fg' : 'fg.muted'}
        cursor={isLocked ? 'not-allowed' : undefined}
        display="flex"
        flex="1"
        flexDirection="column"
        fontSize="md"
        fontStyle={isFuture ? 'italic' : undefined}
        minH="6"
        minW="0"
        px="2"
        py="1"
        rounded="sm"
        textAlign="left"
        type="button"
        _hover={isCurrent || isLocked ? undefined : STEP_HOVER_PROPS}
        onClick={jump}
      >
        <Text as="span" maxW="full" truncate>
          {label}
        </Text>
        {note ? (
          <Text as="span" color="fg.muted" fontSize="xs" fontStyle="normal">
            {note}
          </Text>
        ) : null}
      </chakra.button>
    </chakra.li>
  );
};
