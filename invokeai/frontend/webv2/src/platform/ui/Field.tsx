import type { FeatureHintId } from '@platform/ui/hints';
import type { ReactNode } from 'react';

import {
  chakra,
  Field as ChakraField,
  HStack,
  Stack,
  Text,
  useFieldContext,
  useRecipe,
  type StackProps,
} from '@chakra-ui/react';
import { FEATURE_HINT_FIELD_PROPS, FeatureHint } from '@platform/ui/hints';
import { fieldLabelRecipe } from '@theme/recipes';
import { createContext, useContext, useMemo } from 'react';

// Field context carries a label id whether or not a label renders; raw `Field.Root` hosts render none.
const RenderedLabelIdContext = createContext<string | null>(null);

const RenderedLabelIdProvider = ({ children }: { children: ReactNode }) => {
  const field = useFieldContext();

  return <RenderedLabelIdContext value={field?.ids.label ?? null}>{children}</RenderedLabelIdContext>;
};

/**
 * The id of the label naming the nearest field's control, or null when that field renders none. For custom
 * controls that reference the label themselves; Chakra controls read Field context directly. An inside Field renders
 * no label of its own: a control that shows it takes the id from `useInsideFieldLabel` along with the label.
 */
export const useFieldLabelId = (): string | null => {
  const field = useFieldContext();
  const labelId = useContext(RenderedLabelIdContext);

  return field && field.ids.label === labelId ? labelId : null;
};

/** Wraps label text only: the hint trigger overwrites its child's ID and would break aria-labelledby on the label. */
const renderLabelText = (label: string, hint: FeatureHintId | undefined): ReactNode =>
  hint ? (
    <FeatureHint hint={hint}>
      <chakra.span>{label}</chakra.span>
    </FeatureHint>
  ) : (
    label
  );

/** What a `labelPlacement="inside"` Field hands its control. */
export interface InsideFieldLabel {
  /** The Field's label id: the control names itself by the element `FieldInsideLabel` renders with it. */
  id: string;
  text: string;
  hint?: FeatureHintId;
}

const InsideLabelContext = createContext<InsideFieldLabel | null>(null);

const InsideLabelProvider = ({
  children,
  hint,
  text,
}: {
  children: ReactNode;
  hint: FeatureHintId | undefined;
  text: string;
}) => {
  const labelId = useFieldContext()?.ids.label;
  const value = useMemo(() => (labelId ? { hint, id: labelId, text } : null), [hint, labelId, text]);

  return <InsideLabelContext value={value}>{children}</InsideLabelContext>;
};

/**
 * The label the nearest field asks its control to show at the start of the control's own box; null when that field
 * shows its label outside, or is a raw `Field.Root` nested in an inside Field.
 */
export const useInsideFieldLabel = (): InsideFieldLabel | null => {
  const field = useFieldContext();
  const label = useContext(InsideLabelContext);

  return field && label && field.ids.label === label.id ? label : null;
};

/** Inset from a control's inner edge, as ScrubberField's label is, so stacked inside labels start at one x. */
export const INSIDE_LABEL_INSET = '3';

const INSIDE_LABEL_CSS = {
  // The field label's type, as ScrubberField's inside label uses: stacked scrubbers and selects read as one set.
  ...fieldLabelRecipe.base,
  flexShrink: 0,
  // The value gives way first; only a label longer than half the box truncates.
  maxW: '50%',
  overflow: 'hidden',
  textOverflow: 'ellipsis',
  whiteSpace: 'nowrap',
} as const;

/**
 * The inside label at the start of a control's box. In a trigger button it is a span the trigger's aria-labelledby
 * points at (a `<label>` may not sit in a button); beside an input, `htmlFor` makes it a native label, so a click on
 * it focuses and opens the input like the rest of the box.
 */
export const FieldInsideLabel = ({ htmlFor, label }: { htmlFor?: string; label: InsideFieldLabel }) => {
  const text = renderLabelText(label.text, label.hint);

  return htmlFor ? (
    <chakra.label css={INSIDE_LABEL_CSS} htmlFor={htmlFor} id={label.id}>
      {text}
    </chakra.label>
  ) : (
    <chakra.span css={INSIDE_LABEL_CSS} id={label.id}>
      {text}
    </chakra.span>
  );
};

export const FieldLabel = ({ children }: { children: ReactNode }) => {
  const recipe = useRecipe({ recipe: fieldLabelRecipe });

  return (
    <Text as="span" css={recipe()}>
      {children}
    </Text>
  );
};

interface FieldBaseProps extends Omit<StackProps, 'disabled' | 'id' | 'readOnly' | 'required' | 'title'> {
  id?: string;
  label: string;
  disabled?: boolean;
  invalid?: boolean;
  readOnly?: boolean;
  required?: boolean;
  /** Validation error, shown in place of `helpText`; also marks the field invalid by default. */
  error?: string | null;
  helpText?: string;
  /** Opens an informational hint card when the label is hovered. */
  hint?: FeatureHintId;
  children: ReactNode;
}

export type FieldProps = FieldBaseProps &
  (
    | {
        /** `outside` (default) stacks the label above the control, or beside it when horizontal. */
        labelPlacement?: 'outside';
        labelEnd?: ReactNode;
        orientation?: 'horizontal' | 'vertical';
      }
    | {
        /**
         * `inside` hands the label to the control, which shows it at the start of its own box: one compact row for
         * dense panels. Help and error text stay below the box. Only Select, Combobox and ModelSelect render it; any
         * other child (Input, NumberInput, ScrubberField) is left without a label or an accessible name.
         */
        labelPlacement: 'inside';
        labelEnd?: never;
        orientation?: never;
      }
  );

/** A labelled form field: an uppercase label stacked above its control, with an optional help/error line below. */
export const Field = ({
  children,
  disabled,
  error,
  helpText,
  hint,
  id,
  invalid,
  label,
  labelEnd,
  labelPlacement = 'outside',
  orientation = 'vertical',
  readOnly,
  required,
  ...rest
}: FieldProps) => {
  const recipe = useRecipe({ recipe: fieldLabelRecipe });
  const isHorizontal = orientation === 'horizontal';
  const isInvalid = invalid ?? Boolean(error);
  const hintFieldProps = hint ? FEATURE_HINT_FIELD_PROPS : undefined;
  const ids = useMemo(
    () =>
      id
        ? {
            errorText: `${id}-error`,
            helperText: `${id}-help`,
            label: `${id}-label`,
          }
        : undefined,
    [id]
  );
  const message = error ? (
    <ChakraField.ErrorText color="fg.error" fontSize="xs" role="alert">
      {error}
    </ChakraField.ErrorText>
  ) : helpText ? (
    <ChakraField.HelperText color="fg.muted" fontSize="xs">
      {helpText}
    </ChakraField.HelperText>
  ) : null;

  if (labelPlacement === 'inside') {
    return (
      <ChakraField.Root
        asChild
        disabled={disabled}
        id={id}
        ids={ids}
        invalid={isInvalid}
        readOnly={readOnly}
        required={required}
        unstyled
      >
        <Stack flex="1" gap="1" minW="0" {...hintFieldProps} {...rest}>
          <InsideLabelProvider hint={hint} text={label}>
            {children}
          </InsideLabelProvider>
          {message}
        </Stack>
      </ChakraField.Root>
    );
  }

  const labelContent = <ChakraField.Label css={recipe()}>{renderLabelText(label, hint)}</ChakraField.Label>;

  if (!isHorizontal) {
    return (
      <ChakraField.Root
        asChild
        disabled={disabled}
        id={id}
        ids={ids}
        invalid={isInvalid}
        readOnly={readOnly}
        required={required}
        unstyled
      >
        <Stack flex="1" gap="1.5" minW="0" {...hintFieldProps} {...rest}>
          <HStack align="center" justify="space-between" minW="0">
            {labelContent}
            {labelEnd}
          </HStack>
          <RenderedLabelIdProvider>{children}</RenderedLabelIdProvider>
          {message}
        </Stack>
      </ChakraField.Root>
    );
  }

  return (
    <ChakraField.Root
      asChild
      disabled={disabled}
      id={id}
      ids={ids}
      invalid={isInvalid}
      readOnly={readOnly}
      required={required}
      unstyled
    >
      <Stack align="flex-start" direction="row" flex="1" gap="1.5" minW="0" {...hintFieldProps} {...rest}>
        <HStack align="center" flexShrink="0" justify="space-between" minH="8" minW="0">
          {labelContent}
          {labelEnd}
        </HStack>
        <Stack flex="1" gap="1.5" minW="0" w="full">
          <RenderedLabelIdProvider>{children}</RenderedLabelIdProvider>
          {message}
        </Stack>
      </Stack>
    </ChakraField.Root>
  );
};
