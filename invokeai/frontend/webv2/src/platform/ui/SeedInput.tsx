/* oxlint-disable react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-jsx-as-prop */
import type { SeedMode, SeedSubmissionPlan } from '@platform/core/seed';
import type { LucideIcon } from 'lucide-react';

import { HStack, Icon, Menu, NumberInput, Portal, Stack, Text, useFieldContext } from '@chakra-ui/react';
import { SEED_MAX, SEED_MODES } from '@platform/core/seed';
import { ChevronDownIcon, DicesIcon, LocateFixedIcon, MinusIcon, PlusIcon, ShuffleIcon } from 'lucide-react';
import { useId, useMemo, type MouseEvent } from 'react';
import { useTranslation } from 'react-i18next';

import { Button, IconButton } from './Button';
import { MenuContent } from './Menu';
import { Tooltip } from './Tooltip';

const SEED_MODE_MENU_POSITIONING = { placement: 'bottom-end' } as const;
const TABULAR_NUMS = { fontVariantNumeric: 'tabular-nums' } as const;

const selectInputText = (event: MouseEvent<HTMLInputElement>) => event.currentTarget.select();
const SEED_MODE_ICONS: Record<SeedMode, LucideIcon> = {
  decrement: MinusIcon,
  fixed: LocateFixedIcon,
  increment: PlusIcon,
  random: ShuffleIcon,
};

export interface SeedModeMenuProps {
  value: SeedMode;
  onChange: (mode: SeedMode) => void;
  /** A second tooltip line for hosts that need to say what one step means (a queued run, not a loop pass). */
  description?: string;
  /** Class for the portaled menu content, for hosts whose key handling must skip it (xyflow's `nokey`). */
  contentClassName?: string;
}

export const SeedModeMenu = ({ contentClassName, description, onChange, value }: SeedModeMenuProps) => {
  const { t } = useTranslation();
  // Share trigger IDs; wrapping Menu.Trigger loses its anchor ref.
  const triggerId = useId();
  const triggerIds = useMemo(() => ({ trigger: triggerId }), [triggerId]);
  const label = t('common.seedMode.label');
  const valueLabel = t(`common.seedMode.${value}`);
  const accessibleName = `${label}: ${valueLabel}`;

  return (
    <Menu.Root ids={triggerIds} positioning={SEED_MODE_MENU_POSITIONING}>
      {/* The visible label may truncate in a dense row; the full name always reads here and in the tooltip. */}
      <Tooltip
        content={
          description ? (
            <Stack gap="0.5">
              <Text>{accessibleName}</Text>
              <Text color="fg.subtle">{description}</Text>
            </Stack>
          ) : (
            accessibleName
          )
        }
        ids={triggerIds}
      >
        <Menu.Trigger asChild>
          <Button aria-label={accessibleName} flexShrink={0} gap="1" maxW="9rem" minW="0" size="xs" variant="outline">
            <Icon as={SEED_MODE_ICONS[value]} boxSize="3.5" color="fg.muted" flexShrink={0} />
            <Text as="span" minW="0" truncate>
              {valueLabel}
            </Text>
            <Icon as={ChevronDownIcon} boxSize="3" color="fg.muted" flexShrink={0} />
          </Button>
        </Menu.Trigger>
      </Tooltip>
      <Portal>
        <Menu.Positioner>
          <MenuContent className={contentClassName} minW="16rem">
            <Menu.RadioItemGroup value={value} onValueChange={(event) => onChange(event.value as SeedMode)}>
              {SEED_MODES.map((mode) => (
                <Menu.RadioItem key={mode} py="1.5" value={mode}>
                  {/* The recipe centers the check on the row; on a two-line item it belongs on the label line. */}
                  <Menu.ItemIndicator top="2" transform="none" />
                  <Icon alignSelf="flex-start" as={SEED_MODE_ICONS[mode]} boxSize="3.5" color="fg.subtle" mt="0.5" />
                  <Stack gap="0" minW="0">
                    <Menu.ItemText>{t(`common.seedMode.${mode}`)}</Menu.ItemText>
                    <Text color="fg.subtle" fontSize="2xs">
                      {t(`common.seedMode.${mode}Description`)}
                    </Text>
                  </Stack>
                </Menu.RadioItem>
              ))}
            </Menu.RadioItemGroup>
          </MenuContent>
        </Menu.Positioner>
      </Portal>
    </Menu.Root>
  );
};

export interface SeedSequencePreviewProps {
  /** Id the seed input names in `aria-describedby`. */
  id?: string;
  plan: Pick<SeedSubmissionPlan, 'lastSeed' | 'sequenceLength' | 'startSeed'>;
}

/** Where the next submission's seeds start and end; the host resolves the run count it plans with. */
export const SeedSequencePreview = ({ id, plan }: SeedSequencePreviewProps) => {
  const { t } = useTranslation();

  return (
    <Text color="fg.subtle" css={TABULAR_NUMS} data-testid="seed-sequence-preview" fontSize="2xs" id={id}>
      {plan.sequenceLength > 1
        ? t('common.seedNextBatchRange', { first: plan.startSeed, last: plan.lastSeed })
        : t('common.seedNextBatch', { seed: plan.startSeed })}
    </Text>
  );
};

export interface SeedInputPatch {
  seed?: number;
  seedMode?: SeedMode;
}

export interface SeedInputProps {
  /** Accessible name of the number input; the host renders any visible label. */
  ariaLabel: string;
  /** Absent while the host's field is empty. */
  seed: number | undefined;
  seedMode: SeedMode;
  /** The next submission's stepping plan, resolved by whoever owns the run count; null when the seed holds. */
  plan: SeedSubmissionPlan | null;
  id?: string;
  invalid?: boolean;
  /** Classes on the row, for hosts whose surroundings watch pointer or keys (xyflow's `nodrag nokey`). */
  className?: string;
  description?: string;
  contentClassName?: string;
  onCommit: (patch: SeedInputPatch) => void;
}

/** Random mode disables editing without discarding the displayed seed. */
export const SeedInput = ({
  ariaLabel,
  className,
  contentClassName,
  description,
  id,
  invalid,
  onCommit,
  plan,
  seed,
  seedMode,
}: SeedInputProps) => {
  const { t } = useTranslation();
  const previewId = useId();
  const isRandom = seedMode === 'random';
  // Preserve Field helper/error descriptions alongside the seed preview; error IDs come from Ark input props.
  const field = useFieldContext();
  const describedBy =
    [field?.ariaDescribedby, field?.getInputProps()['aria-errormessage'], plan ? previewId : undefined]
      .filter(Boolean)
      .join(' ') || undefined;
  // Pass host IDs through Zag's ID map; overriding the DOM ID breaks external value synchronization.
  const inputIds = useMemo(() => (id ? { input: id } : undefined), [id]);
  const stepperTranslations = useMemo(
    () => ({ decrementLabel: t('common.decreaseValue'), incrementLabel: t('common.increaseValue') }),
    [t]
  );

  return (
    <Stack className={className} gap="1" w="full">
      <HStack gap="1">
        <NumberInput.Root
          disabled={isRandom}
          flex="1"
          ids={inputIds}
          invalid={invalid}
          max={SEED_MAX}
          min={0}
          minW="0"
          size="xs"
          value={seed === undefined ? '' : String(seed)}
          translations={stepperTranslations}
          onValueChange={({ valueAsNumber }) => {
            // A seed is an integer: a typed fraction rounds rather than reaching the graph.
            if (Number.isFinite(valueAsNumber)) {
              onCommit({ seed: Math.round(valueAsNumber) });
            }
          }}
        >
          <NumberInput.Control />
          <NumberInput.Input
            aria-describedby={describedBy}
            aria-label={ariaLabel}
            css={TABULAR_NUMS}
            onDoubleClick={selectInputText}
          />
        </NumberInput.Root>
        {/* Keep the button's accessible name independent of the tooltip's open state. */}
        <Tooltip content={t('common.newSeed')}>
          <IconButton
            aria-label={t('common.newSeed')}
            color="fg.muted"
            disabled={isRandom}
            flexShrink={0}
            size="xs"
            variant="outline"
            onClick={() => onCommit({ seed: Math.floor(Math.random() * SEED_MAX) })}
          >
            <DicesIcon />
          </IconButton>
        </Tooltip>
        <SeedModeMenu
          contentClassName={contentClassName}
          description={description}
          value={seedMode}
          onChange={(nextMode) => onCommit({ seedMode: nextMode })}
        />
      </HStack>
      {plan ? <SeedSequencePreview id={previewId} plan={plan} /> : null}
    </Stack>
  );
};
