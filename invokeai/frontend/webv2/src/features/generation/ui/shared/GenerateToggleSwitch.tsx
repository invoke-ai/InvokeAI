import type { FeatureHintId } from '@platform/ui/hints';
import type { ReactNode } from 'react';

import { Box, chakra, Flex, Grid, Switch, Text, useRecipe } from '@chakra-ui/react';
import { FEATURE_HINT_FIELD_PROPS, FeatureHint } from '@platform/ui/hints';
import { fieldLabelRecipe } from '@theme/recipes';
import { useCallback, useId, useMemo } from 'react';

const CHECKED_CONTROL = { bg: 'accent.solid' } as const;
// Disabled controls across the app read `not-allowed`; the switch control's recipe already does.
const DISABLED_CURSOR = { cursor: 'not-allowed' } as const;

interface GenerateSwitchControlProps {
  checked: boolean;
  disabled: boolean;
  /** Fixed ids: an ancestor Field would otherwise hand every switch inside it the Field's control id. */
  inputId: string;
  describedBy?: string;
  children?: ReactNode;
  onCheckedChange: (checked: boolean) => void;
}

/** The switch every Generate toggle draws, so size and checked colour cannot drift between its two layouts. */
const GenerateSwitchRoot = ({
  checked,
  children,
  describedBy,
  disabled,
  inputId,
  onCheckedChange,
  ...rootProps
}: GenerateSwitchControlProps & Omit<Switch.RootProps, keyof GenerateSwitchControlProps>) => {
  const ids = useMemo(() => ({ hiddenInput: inputId, label: `${inputId}-label` }), [inputId]);
  // Memoized for the repo's `react-perf/jsx-no-new-function-as-prop` lint, not for render cost.
  const handleChange = useCallback(
    (details: { checked: boolean }) => onCheckedChange(details.checked),
    [onCheckedChange]
  );

  return (
    <Switch.Root checked={checked} disabled={disabled} ids={ids} onCheckedChange={handleChange} {...rootProps}>
      <Switch.HiddenInput aria-describedby={describedBy} />
      {children}
    </Switch.Root>
  );
};

const SwitchControl = () => (
  <Switch.Control _checked={CHECKED_CONTROL}>
    <Switch.Thumb />
  </Switch.Control>
);

interface GenerateToggleSwitchProps {
  checked: boolean;
  disabled?: boolean;
  label: string;
  /** Under the label, wrapping with it; also the switch's description. */
  helpText?: string;
  /** Opens an informational hint card when the label is hovered. */
  hint?: FeatureHintId;
  /** Before the switch, outside its label so a click on it never toggles: a preview of what the switch turns on. */
  preview?: ReactNode;
  /** The control the switch enables, under the row. Clicks on it never reach the switch. */
  children?: ReactNode;
  onCheckedChange: (checked: boolean) => void;
}

/**
 * A Generate setting's on/off row: the label is the switch's own `<label>`, so clicking it toggles and names it. A
 * long label wraps under itself while the switch stays at the trailing edge, level with the label's first line.
 */
export const GenerateToggleSwitch = ({
  checked,
  children,
  disabled = false,
  helpText,
  hint,
  label,
  preview,
  onCheckedChange,
}: GenerateToggleSwitchProps) => {
  const id = useId();
  const labelRecipe = useRecipe({ recipe: fieldLabelRecipe });
  const helpId = helpText ? `${id}-help` : undefined;

  return (
    // The switch root is a `<label>` laid out with `display: contents`, so the help text and preview can take grid
    // cells beside its parts without sitting inside the label, where a click on them would toggle.
    <Grid
      columnGap="3"
      position="relative"
      rowGap="1"
      templateColumns={preview ? 'minmax(0, 1fr) auto auto' : 'minmax(0, 1fr) auto'}
      w="full"
      {...(hint ? FEATURE_HINT_FIELD_PROPS : undefined)}
    >
      <GenerateSwitchRoot
        checked={checked}
        describedBy={helpId}
        disabled={disabled}
        display="contents"
        inputId={id}
        onCheckedChange={onCheckedChange}
      >
        <Switch.Label
          css={labelRecipe()}
          gridRow="1"
          // The arrow, as on every control; `auto` would show a text cursor over the label.
          cursor="default"
          gridColumn="1"
          // The switch height, so the first line sits level with the switch and later lines keep the same rhythm.
          lineHeight="var(--switch-height)"
          overflowWrap="anywhere"
          _disabled={DISABLED_CURSOR}
        >
          {hint ? (
            <FeatureHint hint={hint}>
              <chakra.span>{label}</chakra.span>
            </FeatureHint>
          ) : (
            label
          )}
        </Switch.Label>
        <Flex alignSelf="start" gridColumn="-2" gridRow="1">
          <SwitchControl />
        </Flex>
      </GenerateSwitchRoot>
      {preview ? (
        // Outside the switch root, so `--switch-height` is not in scope: `4` is the default switch's height.
        <Flex align="center" alignSelf="start" gridColumn="2" gridRow="1" h="4">
          {preview}
        </Flex>
      ) : null}
      {helpText ? (
        <Text color="fg.muted" fontSize="xs" gridColumn="1" id={helpId} overflowWrap="anywhere">
          {helpText}
        </Text>
      ) : null}
      {children ? (
        <Box gridColumn="1 / -1" mt="0.5">
          {children}
        </Box>
      ) : null}
    </Grid>
  );
};

/**
 * A switch acting on a list row, named by `label` alone because the row's own title is its visible label and
 * belongs to the row's primary button. Settings rows use GenerateToggleSwitch.
 */
export const GenerateRowSwitch = ({
  checked,
  disabled = false,
  label,
  onCheckedChange,
}: {
  checked: boolean;
  disabled?: boolean;
  label: string;
  onCheckedChange: (checked: boolean) => void;
}) => {
  const id = useId();

  return (
    <GenerateSwitchRoot checked={checked} disabled={disabled} inputId={id} onCheckedChange={onCheckedChange}>
      <SwitchControl />
      <Switch.Label srOnly>{label}</Switch.Label>
    </GenerateSwitchRoot>
  );
};
