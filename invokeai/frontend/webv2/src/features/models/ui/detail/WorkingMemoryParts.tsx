import type { SystemStyleObject } from '@chakra-ui/react';
import type {
  LineDisplayState,
  WorkingMemoryCase,
  WorkingMemoryLine,
  WorkingMemoryLineSize,
  WorkingMemoryPlan,
} from '@features/models/core/workingMemory';

import { Badge, chakra, HStack, Icon, Popover, Portal, Stack, Table, Text } from '@chakra-ui/react';
import { getModelBaseLabel } from '@features/models/core/baseIdentity';
import { describeFact, lineDisplayState, orderContributors, splitLine } from '@features/models/core/workingMemory';
import { formatBytes, formatCount } from '@platform/i18n/languages';
import { IconButton } from '@platform/ui/Button';
import { PopoverContent } from '@platform/ui/Popover';
import { Tooltip, useTooltipTriggerIds } from '@platform/ui/Tooltip';
import { InfoIcon } from 'lucide-react';
import { useTranslation } from 'react-i18next';

const STATE_PALETTES: Record<LineDisplayState, string> = {
  in_use: 'green',
  insufficient: 'gray',
  stale: 'orange',
  stopped: 'gray',
  unusable: 'red',
};

/** One state chip; the color repeats the text, never replaces it. */
export const LineStateBadge = ({ state }: { state: LineDisplayState | null }) => {
  const { t } = useTranslation();
  const label =
    state === 'in_use'
      ? t('models.workingMemory.stateInUse')
      : state === 'stale'
        ? t('models.workingMemory.stateStale')
        : state === 'unusable'
          ? t('models.workingMemory.stateUnusable')
          : state === 'insufficient'
            ? t('models.workingMemory.stateInsufficient')
            : state === 'stopped'
              ? t('models.workingMemory.stateStopped')
              : t('models.workingMemory.stateNone');

  return (
    <Badge
      colorPalette={state ? STATE_PALETTES[state] : 'gray'}
      flexShrink={0}
      fontSize="xs"
      size="lg"
      variant="surface"
    >
      {label}
    </Badge>
  );
};

/** A plan case as "768 × 768 · fp16". */
export const useCaseLabel = () => {
  const { t } = useTranslation();

  return (item: Pick<WorkingMemoryCase, 'height' | 'line' | 'width'>): string =>
    t('models.workingMemory.pair', {
      detail: splitLine(item.line).dtype ?? item.line,
      label: t('models.workingMemory.size', { height: item.height, width: item.width }),
    });
};

/** `decode_fp16` → "Decode · fp16". An operation this version has no name for shows as the server sends it. */
export const useLineLabel = () => {
  const { t } = useTranslation();

  return (line: string): string => {
    const { dtype, operation } = splitLine(line);
    const operationLabel =
      operation === 'decode'
        ? t('models.workingMemory.operationDecode')
        : operation === 'encode'
          ? t('models.workingMemory.operationEncode')
          : operation;

    return dtype ? t('models.workingMemory.pair', { detail: dtype, label: operationLabel }) : operationLabel;
  };
};

const MEGAPIXEL = 1_000_000;

const formatMegapixels = (pixels: number): string => formatCount(Math.round((pixels / MEGAPIXEL) * 100) / 100);

const TABLE_SX: SystemStyleObject = {
  '& td, & th': { borderColor: 'border.subtle', fontSize: 'sm', px: '2', py: '1' },
  '& thead th': {
    color: 'fg.muted',
    fontSize: 'xs',
    fontWeight: '700',
    letterSpacing: 'wider',
    textTransform: 'uppercase',
  },
  '& tbody th': { fontWeight: '400', textAlign: 'start' },
  // Every row starts with its size as a `th`, so each `td` and every header after the first holds a number.
  '& td, & thead th + th': { fontVariantNumeric: 'tabular-nums', textAlign: 'end' },
};

/** A changed environment fact by a readable name; see `describeFact`. */
const useFactLabel = () => {
  const { t } = useTranslation();

  return (key: string): string => {
    const fact = describeFact(key);

    if ('name' in fact) {
      return fact.name;
    }

    switch (fact.translated) {
      case 'allocator':
        return t('models.workingMemory.facts.allocator');
      case 'driver':
        return t('models.workingMemory.facts.driver');
      case 'invokeai':
        return t('models.workingMemory.facts.invokeai');
      case 'osBuild':
        return t('models.workingMemory.facts.osBuild');
      case 'sdpaGuard':
        return t('models.workingMemory.facts.sdpaGuard');
      case 'torchBuild':
        return t('models.workingMemory.facts.torchBuild');
    }
  };
};

/** The state help of a line: why it applies or not, naming the facts that changed or differ. */
const LineStateHelp = ({ line }: { line: WorkingMemoryLine }) => {
  const { t } = useTranslation();
  const factLabel = useFactLabel();
  const facts = line.changed_facts.map(factLabel).join(', ');
  // The server explains a line only stopped attempts measured in English; this says the same in the user's language.
  const text =
    lineDisplayState(line) === 'stopped'
      ? t('models.workingMemory.helpStopped')
      : line.state === 'in_use'
        ? t('models.workingMemory.helpInUse')
        : line.state === 'stale'
          ? t('models.workingMemory.helpStale', { facts })
          : line.state === 'unusable'
            ? facts
              ? t('models.workingMemory.helpUnusable', { facts })
              : t('models.workingMemory.helpUnusableNoFacts')
            : (line.problem ?? t('models.workingMemory.helpInsufficient'));

  return (
    <Text color="fg.muted" fontSize="xs">
      {text}
    </Text>
  );
};

const HINT_POSITIONING = { placement: 'top' } as const;

/** A column header with an info button after its label. */
const HintedColumnHeader = ({ hint, label }: { hint: string; label: string }) => {
  const { t } = useTranslation();
  const name = t('models.workingMemory.aboutColumn', { column: label });
  // The button stays on the line of the label's last word when a narrow column wraps the label.
  const split = label.lastIndexOf(' ') + 1;

  return (
    <Table.ColumnHeader>
      {label.slice(0, split)}
      <chakra.span whiteSpace="nowrap">
        {label.slice(split)}
        <HintButton hint={hint} name={name} />
      </chakra.span>
    </Table.ColumnHeader>
  );
};

/** The explanation opens in a popover, by click or keyboard, so it does not depend on hover. */
const HintButton = ({ hint, name }: { hint: string; name: string }) => {
  const ids = useTooltipTriggerIds();

  return (
    <Popover.Root ids={ids} lazyMount positioning={HINT_POSITIONING} unmountOnExit>
      <Tooltip content={name} ids={ids}>
        <Popover.Trigger asChild>
          <IconButton
            aria-label={name}
            color="fg.subtle"
            h="5"
            minW="5"
            ms="0.5"
            size="sm"
            variant="ghost"
            verticalAlign="middle"
            w="5"
          >
            <Icon as={InfoIcon} boxSize="3" />
          </IconButton>
        </Popover.Trigger>
      </Tooltip>
      <Portal>
        <Popover.Positioner>
          <PopoverContent aria-label={name} maxW="18rem" p="3">
            <Text fontSize="xs">{hint}</Text>
          </PopoverContent>
        </Popover.Positioner>
      </Portal>
    </Popover.Root>
  );
};

/** Why a measured size does not feed the line, or null for a size that does. */
const useNotCountedLabel = () => {
  const { t } = useTranslation();

  return (size: Pick<WorkingMemoryLineSize, 'counted' | 'excluded_reason'>): string | null => {
    if (size.counted) {
      return null;
    }

    switch (size.excluded_reason) {
      case 'fragmented':
        return t('models.workingMemory.notCountedFragmented');
      case 'partial_residency':
        return t('models.workingMemory.notCountedPartial');
      case 'aborted_attempt':
        return t('models.workingMemory.notCountedAborted');
      case null:
        return t('models.workingMemory.notCounted');
    }
  };
};

/**
 * One estimator line: its state, the models whose measurements it pools (named when another model is among them,
 * this model first), the range the user line was fitted on, and every measured size.
 */
export const LineMeasurements = ({ line, modelKey }: { line: WorkingMemoryLine; modelKey: string }) => {
  const { t } = useTranslation();
  const lineLabel = useLineLabel()(line.line);
  const notCountedLabel = useNotCountedLabel();
  const isShared = line.measured_by.some((entry) => entry.model_key !== modelKey);
  const contributors = orderContributors(line.measured_by, modelKey)
    .map((entry) => entry.name ?? t('models.workingMemory.removedModel'))
    .join(', ');

  return (
    <Stack gap="1.5" minW="0">
      <HStack gap="2" wrap="wrap">
        <Text as="h4" fontSize="sm" fontWeight="600">
          {lineLabel}
        </Text>
        <LineStateBadge state={lineDisplayState(line)} />
      </HStack>
      <LineStateHelp line={line} />
      {isShared ? (
        <Text color="fg.muted" fontSize="xs" overflowWrap="anywhere">
          {t('models.workingMemory.measuredBy', { models: contributors })}
        </Text>
      ) : null}
      {line.measured_range ? (
        <Text color="fg.muted" fontSize="xs">
          {t('models.workingMemory.measuredRange', {
            max: formatMegapixels(line.measured_range.max_pixels),
            min: formatMegapixels(line.measured_range.min_pixels),
          })}
        </Text>
      ) : null}
      {line.sizes.length > 0 ? (
        <chakra.div overflowX="auto">
          <Table.Root
            aria-label={t('models.workingMemory.tableLabel', { line: lineLabel })}
            css={TABLE_SX}
            // Fixed columns line the tables of a section up, whatever a row header's note adds; narrow panes scroll.
            minW="26rem"
            tableLayout="fixed"
            variant="line"
          >
            <Table.ColumnGroup>
              <Table.Column w="34%" />
              <Table.Column w="22%" />
              <Table.Column w="22%" />
              <Table.Column w="22%" />
            </Table.ColumnGroup>
            <Table.Header>
              <Table.Row>
                <Table.ColumnHeader>{t('models.workingMemory.columnSize')}</Table.ColumnHeader>
                <HintedColumnHeader
                  hint={t('models.workingMemory.hintMeasured')}
                  label={t('models.workingMemory.columnMeasured')}
                />
                <HintedColumnHeader
                  hint={t('models.workingMemory.hintShipped')}
                  label={t('models.workingMemory.columnShipped')}
                />
                <HintedColumnHeader
                  hint={t('models.workingMemory.hintCalibrated')}
                  label={t('models.workingMemory.columnCalibrated')}
                />
              </Table.Row>
            </Table.Header>
            <Table.Body>
              {line.sizes.map((size) => {
                // A size the fit leaves out stays listed, muted, and names why in its row header.
                const notCounted = notCountedLabel(size);
                const color = notCounted ? 'fg.muted' : undefined;

                return (
                  <Table.Row key={`${String(size.width)}x${String(size.height)}`}>
                    <Table.Cell as="th" color={color} scope="row" whiteSpace="nowrap">
                      {t('models.workingMemory.size', { height: size.height, width: size.width })}
                      {notCounted ? (
                        <chakra.span display="block" fontSize="xs" whiteSpace="normal">
                          {notCounted}
                        </chakra.span>
                      ) : null}
                    </Table.Cell>
                    <Table.Cell color={color} whiteSpace="nowrap">
                      {formatBytes(size.measured_bytes)}
                    </Table.Cell>
                    <Table.Cell color={color} whiteSpace="nowrap">
                      {formatBytes(size.shipped_bytes)}
                    </Table.Cell>
                    <Table.Cell color={color} whiteSpace="nowrap">
                      {formatBytes(size.user_bytes)}
                    </Table.Cell>
                  </Table.Row>
                );
              })}
            </Table.Body>
          </Table.Root>
        </chakra.div>
      ) : null}
    </Stack>
  );
};

/** Other installed models whose estimates the same lines price, so a calibration moves them too. */
export const SharedModelsList = ({ models }: { models: WorkingMemoryPlan['shared_with'] }) => {
  const { t } = useTranslation();
  const lineLabel = useLineLabel();

  return (
    <Stack gap="1">
      <Text as="h4" fontSize="sm" fontWeight="600">
        {t('models.workingMemory.sharedWith', { count: models.length })}
      </Text>
      <chakra.ul display="flex" flexDirection="column" gap="0.5" listStyleType="none" m="0" p="0">
        {models.map((model) => (
          <chakra.li key={model.key} color="fg.muted" fontSize="xs" overflowWrap="anywhere">
            {t('models.workingMemory.pair', {
              detail: [getModelBaseLabel(model.base), ...model.lines.map(lineLabel)].join(', '),
              label: model.name,
            })}
          </chakra.li>
        ))}
      </chakra.ul>
    </Stack>
  );
};
