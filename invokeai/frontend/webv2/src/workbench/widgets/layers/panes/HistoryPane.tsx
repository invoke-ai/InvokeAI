import { chakra, Flex, HStack, Icon, Stack, Text } from '@chakra-ui/react';
import { IconButton } from '@platform/ui/Button';
import { Scrollable } from '@platform/ui/Scrollable';
import { Tooltip } from '@platform/ui/Tooltip';
import { useCanvasEngine, type CanvasEngineHandle } from '@workbench/canvas-operations/react';
import { useCanvasCanRedo, useCanvasCanUndo, useCanvasHistoryEpoch } from '@workbench/widgets/canvas/engineStoreHooks';
import { Redo2Icon, Undo2Icon } from 'lucide-react';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const STEP_HOVER_PROPS = { bg: 'bg.muted', color: 'fg' };

/**
 * Mirror bounded engine history oldest-first; selecting a step replays undo/redo, with future steps below the
 * current position.
 */
export const HistoryPane = () => {
  const { t } = useTranslation();
  const engine = useCanvasEngine();

  if (!engine) {
    return (
      <Flex align="center" color="fg.muted" fontSize="xs" h="full" justify="center" p="4">
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
  const { future, past } = engine.history.getEntries();
  const undo = useCallback(() => engine.history.undo(), [engine]);
  const redo = useCallback(() => engine.history.redo(), [engine]);
  const jump = useCallback((offset: number) => engine.history.stepBy(offset), [engine]);

  return (
    <Flex direction="column" h="full" minH="0">
      <HStack borderBottomWidth="1px" borderColor="border.subtle" flexShrink={0} gap="1" px="2" py="1.5">
        <Tooltip content={t('widgets.canvas.commands.undo')}>
          <IconButton
            aria-label={t('widgets.canvas.commands.undo')}
            color="fg.muted"
            disabled={!canUndo}
            size="2xs"
            variant="ghost"
            onClick={undo}
          >
            <Icon as={Undo2Icon} boxSize="3.5" />
          </IconButton>
        </Tooltip>
        <Tooltip content={t('widgets.canvas.commands.redo')}>
          <IconButton
            aria-label={t('widgets.canvas.commands.redo')}
            color="fg.muted"
            disabled={!canRedo}
            size="2xs"
            variant="ghost"
            onClick={redo}
          >
            <Icon as={Redo2Icon} boxSize="3.5" />
          </IconButton>
        </Tooltip>
        <Text color="fg.muted" fontSize="2xs" fontVariantNumeric="tabular-nums">
          {t('widgets.layers.historyPane.count', { count: past.length })}
        </Text>
      </HStack>
      {past.length === 0 && future.length === 0 ? (
        <Flex align="center" color="fg.muted" flex="1" fontSize="xs" justify="center" p="4">
          {t('widgets.layers.historyPane.empty')}
        </Flex>
      ) : (
        <Scrollable flex="1" minH="0">
          <Stack aria-label={t('widgets.layers.historyPane.steps')} as="ol" gap="0" listStyleType="none" m="0" p="1">
            {past.map((label, index) => (
              <HistoryStep
                key={`past-${index}`}
                isCurrent={index === past.length - 1}
                isFuture={false}
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
                label={label}
                offset={index + 1}
                onJump={jump}
              />
            ))}
          </Stack>
        </Scrollable>
      )}
    </Flex>
  );
};

const HistoryStep = ({
  isCurrent,
  isFuture,
  label,
  offset,
  onJump,
}: {
  isCurrent: boolean;
  isFuture: boolean;
  label: string;
  /** Steps to replay to make this the last applied entry; 0 for the current one. */
  offset: number;
  onJump: (offset: number) => void;
}) => {
  const jump = useCallback(() => {
    if (offset !== 0) {
      onJump(offset);
    }
  }, [offset, onJump]);

  return (
    <chakra.li display="flex" m="0" p="0">
      <chakra.button
        aria-current={isCurrent ? 'step' : undefined}
        bg={isCurrent ? 'bg.emphasized' : 'transparent'}
        // Future (redoable) steps read as dimmed via italics; fg.subtle fails
        // WCAG contrast at this size, so color stays at fg.muted for both.
        color={isCurrent ? 'fg' : 'fg.muted'}
        fontStyle={isFuture ? 'italic' : undefined}
        flex="1"
        fontSize="xs"
        minH="6"
        px="2"
        py="1"
        rounded="sm"
        textAlign="left"
        type="button"
        _hover={isCurrent ? undefined : STEP_HOVER_PROPS}
        onClick={jump}
      >
        <Text as="span" truncate>
          {label}
        </Text>
      </chakra.button>
    </chakra.li>
  );
};
