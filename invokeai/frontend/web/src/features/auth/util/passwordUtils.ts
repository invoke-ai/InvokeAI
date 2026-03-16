export type PasswordStrength = 'weak' | 'moderate' | 'strong';

export type PasswordValidationResult = {
  isValid: boolean;
  message: string;
  strength: PasswordStrength | null;
};

/**
 * Returns the strength level of a password.
 * - weak: less than 8 characters
 * - moderate: 8+ characters but missing uppercase, lowercase, or digit
 * - strong: 8+ characters with uppercase, lowercase, and digit
 */
export const getPasswordStrength = (password: string): PasswordStrength => {
  if (password.length < 8) {
    return 'weak';
  }
  const hasUpper = /[A-Z]/.test(password);
  const hasLower = /[a-z]/.test(password);
  const hasDigit = /\d/.test(password);
  if (!hasUpper || !hasLower || !hasDigit) {
    return 'moderate';
  }
  return 'strong';
};

/**
 * Validates a password field.
 *
 * In strict mode, passwords must be 8+ characters with uppercase, lowercase, and digits.
 * In non-strict mode, any non-empty password is accepted but strength is reported.
 *
 * @param password - The password to validate
 * @param t - Translation function
 * @param strictPasswordChecking - Whether to enforce strict requirements
 * @param allowEmpty - When true, an empty string is treated as "no change" (valid with no message)
 */
export const validatePasswordField = (
  password: string,
  t: (key: string) => string,
  strictPasswordChecking: boolean,
  allowEmpty = false
): PasswordValidationResult => {
  if (password.length === 0) {
    return { isValid: allowEmpty, message: '', strength: null };
  }

  const strength = getPasswordStrength(password);

  if (!strictPasswordChecking) {
    return {
      isValid: true,
      message: t(`auth.passwordStrength.${strength}`),
      strength,
    };
  }

  // Strict mode
  if (password.length < 8) {
    return { isValid: false, message: t('auth.setup.passwordTooShort'), strength };
  }
  const hasUpper = /[A-Z]/.test(password);
  const hasLower = /[a-z]/.test(password);
  const hasDigit = /\d/.test(password);
  if (!hasUpper || !hasLower || !hasDigit) {
    return { isValid: false, message: t('auth.setup.passwordMissingRequirements'), strength };
  }
  return { isValid: true, message: '', strength };
};
