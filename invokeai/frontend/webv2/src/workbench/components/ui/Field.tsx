import type { ReactNode } from 'react';

import { Stack, Text, useRecipe, type StackProps } from '@chakra-ui/react';
import { fieldLabelRecipe } from '@theme/recipes';

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

export interface FieldProps extends Omit<StackProps, 'title'> {
  label: string;
  /** Validation error, shown in place of `helpText`. Mark the control itself invalid via `aria-invalid`. */
  error?: string | null;
  helpText?: string;
  children: ReactNode;
}

/** A labelled form field: an uppercase label stacked above its control, with an optional help/error line below. */
export const Field = ({ children, error, helpText, label, ...rest }: FieldProps) => (
  <Stack flex="1" gap="1.5" minW="0" {...rest}>
    <FieldLabel>{label}</FieldLabel>
    {children}
    {error ? (
      <Text color="fg.error" fontSize="2xs" role="alert">
        {error}
      </Text>
    ) : helpText ? (
      <Text color="fg.subtle" fontSize="2xs">
        {helpText}
      </Text>
    ) : null}
  </Stack>
);
