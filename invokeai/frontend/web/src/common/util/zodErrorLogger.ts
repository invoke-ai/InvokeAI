import type { z } from 'zod';

interface ZodErrorLogOptions {
  context?: string;
}

/**
 * Logs Zod validation errors with a simple description and complete error data.
 */
export const logZodError = (error: z.ZodError, options: ZodErrorLogOptions = {}): void => {
  const { context = 'Validation' } = options;

  // eslint-disable-next-line no-console
  console.error(`${context} failed with ${error.issues.length} errors:`, {
    summary: error.issues.map((issue) => `${issue.path.join('.') || 'root'}: ${issue.message}`),
    issues: error.issues,
  });
};

/**
 * Creates a human-readable summary of Zod errors suitable for user-facing messages.
 * Limits the number of errors shown and provides a count of remaining errors.
 */
export const createZodErrorSummary = (
  error: z.ZodError,
  maxErrors: number = 10,
  pathFormatter?: (path: string, message: string) => string
): string => {
  const defaultFormatter = (path: string, message: string) => `${path || 'Root'}: ${message}`;

  const formatter = pathFormatter || defaultFormatter;

  const errors = error.issues.map((issue) => formatter(issue.path.join('.'), issue.message));

  const visibleErrors = errors.slice(0, maxErrors);
  const remainingCount = errors.length - maxErrors;

  let summary = visibleErrors.join('; ');

  if (remainingCount > 0) {
    summary += ` (+${remainingCount} more error${remainingCount === 1 ? '' : 's'})`;
  }

  return summary;
};
