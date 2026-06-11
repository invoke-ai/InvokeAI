import { useCallback, useRef, useState } from 'react';
import type { z } from 'zod';

/**
 * Minimal zod-backed form state. Validation runs on submit; editing a field
 * that shows an error clears that error immediately (optimistic — the next
 * submit re-validates), so users are not nagged mid-correction.
 */

export interface ZodFormField<Value> {
  value: Value;
  error: string | null;
}

export interface ZodForm<Schema extends z.ZodType<Record<string, unknown>>> {
  values: z.infer<Schema>;
  errors: Partial<Record<keyof z.infer<Schema>, string>>;
  /** Form-level error (refinements that do not map to a single field). */
  formError: string | null;
  isSubmitting: boolean;
  setValue: <Key extends keyof z.infer<Schema>>(key: Key, value: z.infer<Schema>[Key]) => void;
  setValues: (values: z.infer<Schema>) => void;
  /** Validate and run `onValid`; submission errors land in `formError`. */
  handleSubmit: (onValid: (values: z.infer<Schema>) => Promise<void> | void) => Promise<void>;
  reset: (values?: z.infer<Schema>) => void;
}

export const useZodForm = <Schema extends z.ZodType<Record<string, unknown>>>(
  schema: Schema,
  initialValues: z.infer<Schema>
): ZodForm<Schema> => {
  type Values = z.infer<Schema>;

  const [values, setValuesState] = useState<Values>(initialValues);
  const initialValuesRef = useRef(initialValues);
  const [errors, setErrors] = useState<Partial<Record<keyof Values, string>>>({});
  const [formError, setFormError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const validate = useCallback(
    (candidate: Values): { ok: true; data: Values } | { ok: false } => {
      const result = schema.safeParse(candidate);

      if (result.success) {
        setErrors({});
        setFormError(null);

        return { data: result.data as Values, ok: true };
      }

      const nextErrors: Partial<Record<keyof Values, string>> = {};
      let nextFormError: string | null = null;

      for (const issue of result.error.issues) {
        const key = issue.path[0];

        if (typeof key === 'string' && nextErrors[key as keyof Values] === undefined) {
          nextErrors[key as keyof Values] = issue.message;
        } else if (issue.path.length === 0 && nextFormError === null) {
          nextFormError = issue.message;
        }
      }

      setErrors(nextErrors);
      setFormError(nextFormError);

      return { ok: false };
    },
    [schema]
  );

  const setValue = useCallback(<Key extends keyof Values>(key: Key, value: Values[Key]) => {
    setValuesState((currentValues) => ({ ...currentValues, [key]: value }));
    // Editing an errored field clears its error; the next submit re-validates.
    setErrors((currentErrors) => {
      if (currentErrors[key] === undefined) {
        return currentErrors;
      }

      const { [key]: _removed, ...rest } = currentErrors;

      return rest as Partial<Record<keyof Values, string>>;
    });
  }, []);

  const setValues = useCallback((nextValues: Values) => {
    setValuesState(nextValues);
  }, []);

  const reset = useCallback((nextValues?: Values) => {
    setValuesState(nextValues ?? initialValuesRef.current);
    setErrors({});
    setFormError(null);
  }, []);

  const handleSubmit = useCallback(
    async (onValid: (validValues: Values) => Promise<void> | void) => {
      const result = validate(values);

      if (!result.ok) {
        return;
      }

      setIsSubmitting(true);
      setFormError(null);

      try {
        await onValid(result.data);
      } catch (error) {
        setFormError(error instanceof Error ? error.message : 'Something went wrong.');
      } finally {
        setIsSubmitting(false);
      }
    },
    [validate, values]
  );

  return { errors, formError, handleSubmit, isSubmitting, reset, setValue, setValues, values };
};
