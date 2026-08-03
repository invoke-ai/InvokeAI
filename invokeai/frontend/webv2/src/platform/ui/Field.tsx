import type { FeatureHintId } from '@platform/ui/hints';
import type { ReactNode } from 'react';

import { chakra, Field as ChakraField, HStack, Stack, Text, useRecipe, type StackProps } from '@chakra-ui/react';
import { FeatureHint } from '@platform/ui/hints';
import { fieldLabelRecipe } from '@theme/recipes';
import { useMemo } from 'react';

/**
 * The shared, theme-aware uppercase field label. Backed by `fieldLabelRecipe` so
 * every form across the workbench renders an identical label without repeating
 * the same five style props inline.
 */
export const FieldLabel = ({ children }: { children: ReactNode }) => {
  const recipe = useRecipe({ recipe: fieldLabelRecipe });

  return (
    <Text as="span" css={recipe()}>
      {children}
    </Text>
  );
};

export interface FieldProps extends Omit<StackProps, 'disabled' | 'id' | 'readOnly' | 'required' | 'title'> {
  id?: string;
  label: string;
  labelEnd?: ReactNode;
  disabled?: boolean;
  orientation?: 'horizontal' | 'vertical';
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
  orientation = 'vertical',
  readOnly,
  required,
  ...rest
}: FieldProps) => {
  const recipe = useRecipe({ recipe: fieldLabelRecipe });
  const isHorizontal = orientation === 'horizontal';
  const isInvalid = invalid ?? Boolean(error);
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
  // The hint wraps the label's *text*, never the `ChakraField.Label` itself:
  // the hover-card trigger stamps its own `id` on whatever element it is given,
  // which would clobber the id that every control's `aria-labelledby` points at
  // and leave the control with no accessible name. A conditional render, not a
  // conditional hook — `FeatureHint` owns the context read and returns its child
  // untouched when hints are off.
  const labelContent = (
    <ChakraField.Label css={recipe()}>
      {hint ? (
        <FeatureHint hint={hint}>
          <chakra.span>{label}</chakra.span>
        </FeatureHint>
      ) : (
        label
      )}
    </ChakraField.Label>
  );
  const message = error ? (
    <ChakraField.ErrorText color="fg.error" fontSize="2xs" role="alert">
      {error}
    </ChakraField.ErrorText>
  ) : helpText ? (
    <ChakraField.HelperText color="fg.muted" fontSize="2xs">
      {helpText}
    </ChakraField.HelperText>
  ) : null;

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
        <Stack flex="1" gap="1.5" minW="0" {...rest}>
          <HStack align="center" justify="space-between" minW="0">
            {labelContent}
            {labelEnd}
          </HStack>
          {children}
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
      <Stack align="flex-start" direction="row" flex="1" gap="1.5" minW="0" {...rest}>
        <HStack align="center" flexShrink="0" justify="space-between" minH="8" minW="0">
          {labelContent}
          {labelEnd}
        </HStack>
        <Stack flex="1" gap="1.5" minW="0" w="full">
          {children}
          {message}
        </Stack>
      </Stack>
    </ChakraField.Root>
  );
};
