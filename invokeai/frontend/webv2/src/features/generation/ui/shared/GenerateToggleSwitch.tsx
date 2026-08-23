/* oxlint-disable react-perf/jsx-no-new-object-as-prop, react-perf/jsx-no-new-function-as-prop */
import { Switch } from '@chakra-ui/react';
import { useId } from 'react';

interface GenerateToggleSwitchProps {
  checked: boolean;
  disabled?: boolean;
  label: string;
  /** Render the label beside the control; otherwise it is screen-reader only. */
  labelVisible?: boolean;
  onCheckedChange: (checked: boolean) => void;
}

/**
 * The one switch used across the Generate panel. Always owns its hidden-input
 * ids: sibling switches inside a shared Field.Root would otherwise receive the
 * same control id and a label click would toggle the wrong one.
 */
export const GenerateToggleSwitch = ({
  checked,
  disabled = false,
  label,
  labelVisible = false,
  onCheckedChange,
}: GenerateToggleSwitchProps) => {
  const id = useId();

  return (
    <Switch.Root
      checked={checked}
      disabled={disabled}
      ids={{ hiddenInput: id, label: `${id}-label` }}
      size="sm"
      onCheckedChange={(event) => onCheckedChange(event.checked)}
    >
      <Switch.HiddenInput />
      <Switch.Control _checked={{ bg: 'accent.solid' }}>
        <Switch.Thumb />
      </Switch.Control>
      <Switch.Label fontSize="xs" srOnly={!labelVisible}>
        {label}
      </Switch.Label>
    </Switch.Root>
  );
};
