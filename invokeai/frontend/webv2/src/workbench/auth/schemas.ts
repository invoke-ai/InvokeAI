import { z } from 'zod';

/**
 * Form schemas for the auth surfaces. Password rules mirror the backend's
 * `validate_password_strength`: 8+ characters with an uppercase letter, a
 * lowercase letter, and a digit. When strict checking is off the backend
 * accepts any non-empty password, so the schemas relax to match.
 */

export type PasswordStrength = 'weak' | 'moderate' | 'strong';

export const getPasswordStrength = (password: string): PasswordStrength => {
  if (password.length < 8) {
    return 'weak';
  }

  const hasUpper = /[A-Z]/.test(password);
  const hasLower = /[a-z]/.test(password);
  const hasDigit = /\d/.test(password);

  return hasUpper && hasLower && hasDigit ? 'strong' : 'moderate';
};

export const PASSWORD_RULES_HINT = 'At least 8 characters, with an uppercase letter, a lowercase letter, and a digit.';

/**
 * Deliberately lenient: the backend accepts special-use domains (`@localhost`,
 * `.local`) for development setups, so the form only catches obvious mistakes
 * and leaves real validation to the server.
 */
const emailSchema = z
  .string()
  .trim()
  .refine((value) => {
    const atIndex = value.indexOf('@');

    return atIndex > 0 && atIndex < value.length - 1;
  }, 'Enter a valid email address.');

const createPasswordSchema = (strict: boolean) =>
  strict
    ? z
        .string()
        .min(8, 'Password must be at least 8 characters long.')
        .refine((value) => getPasswordStrength(value) === 'strong', PASSWORD_RULES_HINT)
    : z.string().min(1, 'Enter a password.');

/** Optional-password variant: empty string means "leave unchanged". */
const createOptionalPasswordSchema = (strict: boolean) =>
  z.string().refine((value) => value === '' || !strict || getPasswordStrength(value) === 'strong', PASSWORD_RULES_HINT);

export const loginSchema = z.object({
  email: emailSchema,
  password: z.string().min(1, 'Enter your password.'),
  rememberMe: z.boolean(),
});

export type LoginFormValues = z.infer<typeof loginSchema>;

export const createSetupSchema = (strict: boolean) =>
  z
    .object({
      confirmPassword: z.string(),
      displayName: z.string(),
      email: emailSchema,
      password: createPasswordSchema(strict),
    })
    .refine((values) => values.password === values.confirmPassword, {
      message: 'Passwords do not match.',
      path: ['confirmPassword'],
    });

export type SetupFormValues = z.infer<ReturnType<typeof createSetupSchema>>;

export const createProfileSchema = (strict: boolean) =>
  z
    .object({
      confirmPassword: z.string(),
      currentPassword: z.string(),
      displayName: z.string(),
      newPassword: createOptionalPasswordSchema(strict),
    })
    .refine((values) => values.newPassword === '' || values.currentPassword !== '', {
      message: 'Enter your current password to set a new one.',
      path: ['currentPassword'],
    })
    .refine((values) => values.newPassword === '' || values.newPassword === values.confirmPassword, {
      message: 'Passwords do not match.',
      path: ['confirmPassword'],
    });

export type ProfileFormValues = z.infer<ReturnType<typeof createProfileSchema>>;

/**
 * One shape serves both admin user-form modes. Creating requires email and
 * password; editing ignores the email field and treats an empty password as
 * "leave unchanged".
 */
export const createUserFormSchema = (strict: boolean, requireCredentials: boolean) =>
  z.object({
    displayName: z.string(),
    email: requireCredentials ? emailSchema : z.string(),
    isAdmin: z.boolean(),
    password: requireCredentials ? createPasswordSchema(strict) : createOptionalPasswordSchema(strict),
  });

export type UserFormValues = z.infer<ReturnType<typeof createUserFormSchema>>;
